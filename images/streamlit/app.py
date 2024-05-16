import re
import random
import logging
import datetime
import collections as cl

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

import matplotlib as mp
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from wordcloud import WordCloud
from nltk.corpus import stopwords

from sqlalchemy import select, desc, and_, func, cast, Integer, create_engine

import utils as ut
from video_text2text import VideoSummary, VideoSuggestions, ClusterShortName
from models import (
    Video, Comment, Cluster,
    CommenterStats, VideoStats,
    get_db
)


logger = logging.getLogger(__name__)


DB_PATH = 'data/youtube.db'
SUMMARY_SUGGEST_CACHE_DIR = './data/summary-suggest-gpt'


def stmt_to_pandas(stmt, index_col=None):
    db_factory = get_db(DB_PATH)

    with db_factory() as db_session:
        result = db_session.execute(stmt)
        result = pd.DataFrame(result.fetchall(), columns=result.keys())

    if index_col is not None:
        result.set_index(index_col, inplace=True)
        assert result.index.is_unique

    return result


def get_summary(db_session, key=None):
    return VideoSummary(
        session=db_session,
        filters={'is_full_documentary': True},
        cache_dir=SUMMARY_SUGGEST_CACHE_DIR
    ).text2text(key)


def get_suggest(db_session, key=None):
    return VideoSuggestions(
        session=db_session,
        filters={'is_full_documentary': True},
        cache_dir=SUMMARY_SUGGEST_CACHE_DIR
    ).text2text(key)


def get_cluster_shortname(db_session, key):
    return ClusterShortName(
        session=db_session,
        cache_dir=SUMMARY_SUGGEST_CACHE_DIR
    ).text2text(key)


@st.cache_data
def get_video_stats():
    ret = stmt_to_pandas(
        select(
            VideoStats.video_id.label('video_id'),
            VideoStats.comment_cnt.label('comment_cnt'),
            VideoStats.update_cnt.label('update_cnt'),
            VideoStats.view_cnt.label('view_cnt'),
            VideoStats.avg_sentiment_negative.label('negative'),
            Video.publish_dt.label('publish_dt'),
            Video.short_title.label('short_title'),
            func.replace(func.replace(Video.title, '"', ''), "'", '').label('title'),
        )
        .select_from(VideoStats)
        .join(Video, Video.id == VideoStats.video_id)
        .filter(VideoStats.is_full_documentary),

        index_col='video_id'
    )

    ret['display_title'] = ret['publish_dt'].dt.strftime('%Y-%m-%d') + ': ' + ret['short_title']
    assert ret['display_title'].nunique() == ret.shape[0]

    return ret


@st.cache_data
def get_superfan_commenters():
    return stmt_to_pandas(
        select(
            CommenterStats.commenter_id.label('commenter_id'),
            CommenterStats.display_name.label('display_name'),
            CommenterStats.comment_cnt.label('comment_cnt'),
            CommenterStats.update_cnt.label('update_cnt'),
            (1 + (-1 * CommenterStats.avg_sentiment_negative)).label('avg_sentiment'),
        )
        .select_from(CommenterStats)
        .filter(
            CommenterStats.is_full_documentary,
            (CommenterStats.comment_cnt > 200),
        )
        .order_by(desc('avg_sentiment'))
        .limit(10),

        index_col='commenter_id'
    )


@st.cache_data
def get_as_of_date():
    return stmt_to_pandas(
        select(
            func.max(VideoStats.last_comment_dt)
        )
        .select_from(VideoStats)
    ).iloc[0, 0]


@st.cache_data
def video_forecasts():
    period_len = 30
    cutoff = get_as_of_date() - datetime.timedelta(days=period_len)

    df = stmt_to_pandas(
        select(
            Comment.video_id.label('video_id'),
            (
                func.trunc((
                    func.julianday(Comment.publish_dt) -
                    func.julianday(Video.publish_dt)
                ) / period_len)
            ).label('relperiod'),
            func.max(Video.publish_dt).label('video_publish_dt'),

            func.sum(cast(Comment.is_update, Integer)).label('updates'),
            func.avg(Comment.sentiment_negative).label('sentiment'),
            func.count(Comment.id).label('comments'),
        )
        .select_from(Comment)
        .join(Video, Video.id == Comment.video_id)
        .filter(Video.is_full_documentary)
        .filter(Comment.video_id.in_(
            select(
                Comment.video_id
            )
            .select_from(Comment)
            .filter(Comment.publish_dt >= func.date(cutoff.isoformat()))
        ))
        .group_by('video_id', 'relperiod')
    )

    df['last_period'] = (df['relperiod'] == df.groupby('video_id')['relperiod'].transform('max'))

    df['ds'] = df['video_publish_dt'] + df['relperiod'].apply(lambda x: pd.Timedelta(days=x * period_len))
    df.set_index('ds', inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period(f'{period_len}D')

    forecasts = []
    for video_id in df['video_id'].unique():
        video_data = df.loc[df['video_id'] == video_id]

        training_data = video_data.loc[~video_data['last_period']]
        test_data = video_data.loc[video_data['last_period']]

        if training_data.shape[0] < 2:  # only videos >= 3 months old
            continue

        #
        # Updates
        #

        forecast_updates = 0  # hard to beat!

        forecasts += [{
            'metric': 'updates',
            'video_id': video_id,
            'actual': test_data['updates'].item(),
            'forecast': forecast_updates,
        }]

        #
        # Comments
        #

        forecast_comments = ExponentialSmoothing(
            training_data['comments'],
            trend='multiplicative',
            seasonal=None,
            initialization_method='estimated',
            missing='raise',
            use_boxcox=True,
        ).fit().forecast(1).iloc[0]

        forecasts += [{
            'metric': 'comments',
            'video_id': video_id,
            'actual': test_data['comments'].item(),
            'forecast': forecast_comments,
        }]

        #
        # Sentiment
        #

        if test_data['comments'].item() >= 10:
            forecast_sentiment = np.average(
                training_data['sentiment'],
                weights=training_data['comments']
            )

            forecasts += [{
                'metric': 'negative_sentiment',
                'video_id': video_id,
                'actual': test_data['sentiment'].item(),
                'forecast': forecast_sentiment,
            }]

    forecasts = pd.DataFrame(forecasts)

    forecasts['residual'] = forecasts['actual'] - forecasts['forecast']
    forecasts['pct_residual'] = 100 * (forecasts['residual'] / forecasts['forecast'])

    forecasts['abs_residual'] = (forecasts['actual'] - forecasts['forecast']).abs()
    forecasts['pct_abs_residual'] = 100 * (forecasts['abs_residual'] / forecasts['forecast'].abs())

    forecasts.replace({
        'pct_residual': {np.inf: np.nan, -np.inf: np.nan},
        'pct_abs_residual': {np.inf: np.nan, -np.inf: np.nan},
    }, inplace=True)

    return forecasts


@st.cache_data
def get_topic_data():
    ret = stmt_to_pandas(
        select(
            Cluster.id.label('cluster_id'),
            func.count(Cluster.id).label('comment_cnt'),
            func.avg(Comment.sentiment_negative).label('mean_negative'),
            func.stdev(Comment.sentiment_negative).label('std_negative'),
        )
        .select_from(Cluster)
        .join(Comment, Comment.cluster_id == Cluster.id)
        .join(Video, Video.id == Comment.video_id)
        .filter(Video.is_full_documentary)
        .group_by('cluster_id'),
    )

    db_factory = get_db(DB_PATH)
    with db_factory() as db_session:
        tmp = ret['cluster_id'].map(lambda s: get_cluster_shortname(db_session, s))

    ret['description'] = tmp.apply(lambda s: s['response']).str.strip()
    ret['prompt'] = tmp.apply(lambda s: s['prompt']).str.strip()

    ret = ret.loc[ret['description'] != 'no topic']
    ret = ret.loc[ret['cluster_id'] != 'cluster:-1']

    ret['comment_share'] = ret['comment_cnt'] / ret['comment_cnt'].sum()

    # .str.capitalize() lowercases everything else
    ret['description'] = ret['description'].apply(lambda s: s[0].upper() + s[1:])

    ret = ret[['description', 'comment_share', 'mean_negative', 'std_negative', 'prompt']] \

    return ret


def sentiment_bucket(score, baseline):
    if score > 0.8:  # baseline.quantile(0.8):
        # return 'very_positive'
        return '😍'
    elif score > 0.6:  # baseline.quantile(0.6):
        # return 'positive'
        return '😃'
    elif score > 0.4:  # baseline.quantile(0.4):
        # return 'neutral'
        return '😑'
    elif score > 0.2:  # baseline.quantile(0.2):
        # return 'negative'
        return '😦'
    else:  # score < baseline.quantile(0.2):
        # return 'very_negative'
        return '😡'


def comment_counts_over_time(comments, block_hours=3):
    # Function to round/truncate datetime to the nearest block
    def round_to_nearest_block(dt, block_hours):
        rounded_minute = (dt.minute // (block_hours * 60)) * (block_hours * 60)
        return dt.replace(minute=rounded_minute, second=0, microsecond=0)

    # Aggregate comments
    comment_blocks = cl.defaultdict(int)
    for comment in comments:
        block_time = round_to_nearest_block(comment.publish_dt, block_hours)
        comment_blocks[block_time] += 1

    # Create a sorted list of times and counts
    sorted_times = sorted(comment_blocks.keys())
    sorted_counts = [comment_blocks[time] for time in sorted_times]

    # Create DataFrame
    data = pd.DataFrame({'Time Block': sorted_times, 'Comments': sorted_counts})
    data.set_index('Time Block', inplace=True)

    return data


def make_wordcloud(comments):
    swd = stopwords.words('english')
    swd += [
        'going', 'know', 'say', 'one', 'people', 'right', 'need', 'today',
        'make', 'time', 'want', 'help', 'new', 'well', 'way', 'think', 'day',
        'see', 'come', 'go', 'look', 'even', 'u', 'saw', 'said', 'says',
        'dont', 'got', 'thing', 'get', "we're", "i'm", 'like', "that's",
        "they're", "we've", 'coming', 'could', 'would', 'also', 'saying',
        "there's",
        'news', 'radio',
    ]

    wfreq, scores = cl.Counter(), {}
    for comment in comments:
        doc = comment.text
        score = 1 + (-1 * comment.sentiment_negative)

        for word in ut.remove_punctuation(doc).lower().strip().split():
            if word in swd:
                continue

            wfreq[word] += 1
            scores[word] = scores.get(word, 0) + score

    # A word's score is the average score of the sentences where it appears
    for word in scores:
        scores[word] /= wfreq[word]

    # Normalizing the sentiment scores to a range [0, 1]
    min_score = min(scores.values())
    max_score = max(scores.values())
    norm = mp.colors.Normalize(vmin=min_score, vmax=max_score)

    # Creating a custom colormap (you can modify this to your preference)
    colormap = mp.colors.LinearSegmentedColormap.from_list("sentiment_colors", ["red", "white", "blue"])

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        color = colormap(norm(scores[word]))
        return tuple(int(255 * c) for c in color[:3])

    wordcloud = WordCloud(
        background_color='white',
        width=400,
        height=400,
        color_func=color_func
    ).generate_from_frequencies(wfreq)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")

    # Create a ScalarMappable and initialize a data array with the normalized scores
    sm = mp.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Add the color bar to the figure
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label('Sentiment Score', rotation=90)
    cbar.ax.yaxis.set_label_position('right')
    cbar.ax.text(1.3, 0.95, 'Positive', ha='left', va='bottom')
    cbar.ax.text(1.3, 0.08, 'Negative', ha='left', va='top')

    st.pyplot(fig)


def process_llm_response(resp, comments):
    resp = resp.split('\n\n')
    resp = resp[:-1]  # boilerplate?

    summary = resp[0]

    point_pattern = r'\d+\.\s+\*\*(.*?):?\*\*:?\s*\n?\s*(.*)'

    points = {}
    for point in resp[1:]:
        match = re.search(point_pattern, point, re.DOTALL)
        if match:
            title, text = match.groups()
            text = text.strip()

            def replace_with_tooltip(match):
                key = match.group(0)
                comment = comments.get(key, '').replace('\n', ' ').replace('\r', ' ')
                return f'<div class="tooltip">{key}<span class="tooltiptext">{comment}</span></div>'

            pattern = r'\[\d+\]'
            text = re.sub(pattern, replace_with_tooltip, text)

            points[title] = text.strip().lstrip('-').lstrip()

    return summary, points


def display_tooltip_style():
    st.markdown("""
    <style>
        .tooltip {
            position: relative;
            display: inline-block;
            text-decoration: underline dashed; /* Dashed underline */
            color: gray; /* Gray color */
            cursor: pointer; /* Changes the cursor to indicate it's clickable */
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px; /* Adjust width as needed */
            background-color: black;
            color: #fff; /* Ensure text is visible on the background */
            text-align: center;
            border-radius: 6px;
            padding: 5px 0; /* Adjust padding for spacing */

            /* Positioning - adjust these as needed */
            position: absolute;
            z-index: 1;
            bottom: 100%; /* Position above the element */
            left: 50%; /* Center the tooltip */
            margin-left: -100px; /* Shift left by half of the tooltip's width */
        }

        .tooltip:hover .tooltiptext {
            visibility: visible; /* Ensure it becomes visible on hover */
        }
    </style>
    """, unsafe_allow_html=True)


def display_video_change_alerts(n=None):
    video_stats = get_video_stats()

    vf = video_forecasts()
    tv = {
        k: vf.loc[vf['metric'] == k]
             .sort_values('abs_residual', ascending=False)
             .drop('metric', axis=1)
             .set_index('video_id')
        for k in vf['metric'].unique()
    }

    st.header('Video Change Alerts')
    st.markdown('Which videos have had significant recent changes in number '
                'of comments, number of update requests or average sentiment '
                'in the last month?')
    st.markdown('Forecasts are based on:')
    st.markdown('- **Comments**: An [exponential smoothing model](https://www.statsmodels.org/dev/tsa.html#exponential-smoothing) with no '
                'seasonality and a multiplicative trend.')
    st.markdown('- **Sentiment**: The weighted average of sentiment '
                'at the month level, weighting by number of comments per '
                'month. "Sentiment" here means mood or affect -- a comment ' +
                'that briefly expresses agreement with or appreciation for ' +
                'a video about, say, Putin before unloading three ' +
                'paragraphs on how Putin is terrible would have a quite '
                'negative sentiment score.')
    st.markdown('- **Updates**: Assuming exactly 0 update requests for all '
                'videos. These are highly informative but rare enough that '
                'this is still a good baseline.')

    for metric in sorted(tv.keys()):
        expander_title = ' '.join([s.title() for s in metric.split('_')])

        with st.expander(expander_title, expanded=True):
            trending = tv[metric]
            if n is not None:
                trending = trending.head(n)

            trending = trending.merge(
                video_stats[['short_title', 'publish_dt', 'comment_cnt']],
                how='inner',
                left_index=True,
                right_index=True
            )

            trending['publish_dt'] = trending['publish_dt'].dt.strftime('%Y-%m-%d')
            trending['incl_comments'] = tv['comments']['actual'].astype(int)

            cols = ['short_title', 'publish_dt', 'forecast', 'actual',
                    'residual', 'pct_residual', 'abs_residual', 'pct_abs_residual',
                    'incl_comments', 'comment_cnt']
            if metric == 'comments':
                cols = [c for c in cols if c not in ('comment_cnt', 'incl_comments')]
            elif metric == 'updates':
                cols = [c for c in cols if c not in ('pct_residual', 'pct_abs_residual')]

            trending = trending[cols]

            trending.reset_index(drop=True, inplace=True)

            forecast_int = trending['forecast'].apply(float.is_integer).all()
            actual_int = trending['actual'].apply(float.is_integer).all()
            both_int = forecast_int and actual_int

            forecast_fmt = '%d' if forecast_int else '%.2f'
            actual_fmt = '%d' if actual_int else '%.2f'
            residual_fmt = '%d' if both_int else '%.2f'
            abs_residual_fmt = '%d' if both_int else '%.2f'

            with pd.option_context('display.precision', 2):
                st.dataframe(
                    trending,
                    hide_index=True,
                    column_order=cols,
                    column_config={
                        'short_title': st.column_config.TextColumn(label='Title'),
                        'publish_dt': st.column_config.DatetimeColumn(label='Date', format='YYYY-MM-DD'),
                        'forecast': st.column_config.NumberColumn(label='Forecast', format=forecast_fmt),
                        'actual': st.column_config.NumberColumn(label='Actual', format=actual_fmt),
                        'residual': st.column_config.NumberColumn(label='Diff', format=residual_fmt),
                        'pct_residual': st.column_config.NumberColumn(label='% Diff', format='%.1f%%'),
                        'abs_residual': st.column_config.NumberColumn(label='Abs. Diff', format=abs_residual_fmt),
                        'pct_abs_residual': st.column_config.NumberColumn(label='% Abs. Diff', format='%.1f%%'),
                        'incl_comments': st.column_config.NumberColumn(label='Incl. Comments'),
                        'comment_cnt': st.column_config.NumberColumn(label='Total Comments'),
                    }
                )


def display_summary(db_session, key=None):
    summary = get_summary(db_session, key)

    if summary['prompt'] != '':
        theme_comments = VideoSummary._comments_from_prompt(summary['prompt'])
        theme_summary, theme_points = process_llm_response(summary['response'], theme_comments)

        st.header('Themes')
        st.markdown(theme_summary, unsafe_allow_html=True)
        for k, v in theme_points.items():
            with st.expander(k):
                # Need the HTML comment up front to make streamlit recognize
                # this as HTML and not try to examine it, modify it, mess up
                # our HTML tags
                st.markdown('<!-- -->' + v, unsafe_allow_html=True)


def display_suggest(db_session, key=None):
    suggest = get_suggest(db_session, key)

    if suggest['prompt'] != '':
        suggest_comments = VideoSuggestions._comments_from_prompt(suggest['prompt'])
        suggest_summary, suggest_points = process_llm_response(suggest['response'], suggest_comments)

        st.header('Suggestions')
        st.markdown(suggest_summary, unsafe_allow_html=True)
        for k, v in suggest_points.items():
            with st.expander(k):
                # as in display_summary
                st.markdown('<!-- -->' + v, unsafe_allow_html=True)


def display_topics(db_session):
    topic_data = get_topic_data()
    topic_data['comment_share'] *= 100

    topic_data['std_negative_3tile'] = pd.qcut(topic_data['std_negative'], 3, labels=False) \
        .replace({0: '\u2705', 1: '\u2705\u2705', 2: '\u2705\u2705\u2705'})

    st.header('Breakdown: Detailed Topics')
    st.markdown('This is a sortable table of more detailed topics and information about them. You can see')
    st.markdown('- The fraction of comments that each topic accounts for;')
    st.markdown('- How negative comments about the topic are;')
    st.markdown('- How much variance there is in the negativity. Higher numbers ' +
                'mean more positive *and* more negative comments, while lower' +
                'numbers mean a more homogeneous mood for the discussion.')
    st.markdown('Click on a column header to sort by it; click again to change ' +
                'the sort order; click a third time to reset sorting.')

    st.dataframe(topic_data, column_config={
        'description': st.column_config.TextColumn('Topic', width='large', help='Topic/cluster of comments'),
        'comment_share': st.column_config.NumberColumn('Share of Comments', help="Share of all comments which are in this topic", format='%.1f%%'),
        'mean_negative': st.column_config.NumberColumn('Avg. Negativity', help='Average negativity score of comments in this topic', format='%.2f'),
        'std_negative_3tile': st.column_config.TextColumn('Sentiment Diversity'),
        'std_negative': None,
        'prompt': None,
    }, hide_index=True)

    st.subheader('Sample Comments from Each Topic')

    desc = st.selectbox("Select a Topic Description", topic_data['description'])
    row = topic_data[topic_data['description'] == desc].iloc[0]

    with st.expander('Comments for Selected Topic', expanded=False):
        comments = ClusterShortName._comments_from_prompt(row['prompt'])

        keys = list(comments.keys())
        random.shuffle(keys)
        keys = keys[:10]

        for key in keys:
            st.markdown('- ' + comments[key])


def display_sentiment_chart(db_session, video_stats):
    hist_chart = alt.Chart(video_stats).mark_bar().encode(
        alt.X('Sentiment:Q', bin=alt.Bin(maxbins=20)),
        alt.Y('count()'),
    )

    dummy = pd.DataFrame({'mean': [video_stats['Sentiment'].mean()]})
    vline_chart = alt.Chart(dummy).mark_rule(color='red', size=2).encode(x='mean:Q')

    label_chart = alt.Chart(dummy).mark_text(
        align='center',  # Position the text to the left of the line
        baseline='bottom',  # Center vertically
        dy=-5  # Shift text a bit to the right
    ).encode(
        x='mean:Q',
        y=alt.value(5),  # Position the label at the top of the chart
        text=alt.value(f"Avg: {dummy.iloc[0,0]:.2f}")
    )

    chart = hist_chart + vline_chart + label_chart

    st.write("## Video-Level Negative Sentiment Scores")
    st.altair_chart(chart, use_container_width=True)


def display_superfans(db_session):
    dat = get_superfan_commenters().reset_index(drop=True)

    st.header('Superfan Commenters')
    st.markdown('The most positive commenters with >= 200 comments.')
    st.dataframe(
        dat,
        hide_index=True,
        column_config={
            'display_name': st.column_config.TextColumn(label='Commenter'),
            'update_cnt': st.column_config.NumberColumn(label='# Updates', format='%d'),
            'comment_cnt': st.column_config.NumberColumn(label='# Comments', format='%d'),
            'avg_sentiment': st.column_config.NumberColumn(label='Avg. Positivity', format='%.2f'),
        }
    )


def video_card(db_session, video_stats):
    #
    # Topmost video selection dropdowns
    #

    col1, col2 = st.columns(2)
    with col1:
        sort_order = st.selectbox(
            'Sort Videos By',
            options=['Time', 'Alphabetical', 'Comments', 'Updates']
        )

        if sort_order == 'Time':
            opts = video_stats.sort_values('publish_dt', ascending=False)
        elif sort_order == 'Alphabetical':
            opts = video_stats.sort_values('title', ascending=True)
        elif sort_order == 'Updates':
            opts = video_stats.sort_values('update_cnt', ascending=False)
        else:  # sort_order == 'Comments'
            opts = video_stats.sort_values('comment_cnt', ascending=False)
        opts = opts['display_title']
    with col2:
        selected_title = st.selectbox('Select a Video', options=opts)

    video_id = video_stats.loc[video_stats['display_title'] == selected_title].index.item()
    video = db_session.query(Video).filter(Video.id == video_id).one()
    comments = db_session.query(Comment).filter(Comment.video_id == video_id).all()

    last_2_weeks_date = get_as_of_date() - datetime.timedelta(days=14)

    plus_48_hours_date = video.publish_dt + datetime.timedelta(days=2)

    st.markdown('---')

    #
    # Top card with thumbnail
    #

    col1, _, _, col4 = st.columns(4)
    with col1:
        st.image('assets/frontline.png')
    with col4:
        st.image('assets/mit.png')

    st.markdown('---')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(video.thumbnail)
    with col2:
        title = video.short_title
        date = video.publish_dt.strftime('%B %d, %Y')
        views = video.view_cnt
        desc = video.short_description

        st.write(f"**Video:** {title}")
        st.write(f"**Publish Date:** {date}")
        st.write(f"**View Count:** {views:,}")
        st.write(f"**Description:** {desc}")

    st.markdown('<hr style="height:4px;border:none;color:#ff0000;' +
                'background-color:#ff0000;" />', unsafe_allow_html=True)

    #
    # More details
    #

    cols = st.columns(5)
    with cols[0]:
        num_comments = len(comments)

        avg_comments = video_stats['comment_cnt'].mean()
        vs_baseline = (num_comments - avg_comments) / avg_comments
        vs_baseline = int(round(100 * vs_baseline, 0))
        vs_baseline = str(vs_baseline) + '% vs avg'

        st.metric('Comments', f'{num_comments:,}', vs_baseline)
    with cols[1]:
        num_comments = len([c for c in comments if c.publish_dt < plus_48_hours_date])
        st.metric('First 48h', f'{num_comments:,}')
    with cols[2]:
        num_comments = len([c for c in comments if c.publish_dt > last_2_weeks_date])
        st.metric('Last 2w', f'{num_comments:,}')
    with cols[3]:
        num_updates = sum(['update' in c.text.lower() for c in comments])

        avg_updates = video_stats['update_cnt'].mean()
        vs_baseline = (num_updates - avg_updates) / avg_updates
        vs_baseline = int(round(100 * vs_baseline, 0))
        vs_baseline = str(vs_baseline) + '% vs avg'

        st.metric('Update Requests', f'{num_updates:,}', vs_baseline)
    with cols[4]:
        baseline = 1 - video_stats['negative']
        baseline = baseline[baseline.notna()]
        avg_score = baseline.mean()

        score = 1 - pd.Series([c.sentiment_negative for c in comments]).mean()

        if not np.isnan(score):
            vs_baseline = (score - avg_score) / avg_score
            vs_baseline = int(round(100 * vs_baseline, 0))
            vs_baseline = str(vs_baseline) + '% vs avg'

            bucket = sentiment_bucket(score, baseline)
            val = str(round(score, 2)) + '  ' + bucket

            st.metric('Sentiment', val, vs_baseline)
        else:
            st.metric('Sentiment', '--')

    left, right = st.columns(2)
    with left:
        if len(comments) > 0:
            st.subheader('Comments over time')

            cc = comment_counts_over_time(comments)
            st.line_chart(cc)
    with right:
        if len(comments) > 0:
            st.subheader("What's in the comments?")

            make_wordcloud(comments)

    if len(comments) > 0:
        st.subheader('Most liked comments')

        like_comments = sorted(comments, key=lambda s: s.like_cnt,
                               reverse=True)

        like_comments = pd.DataFrame([
            {
                'Author': c.commenter.display_name,
                'Time': c.publish_dt.strftime('%I:%M %p, %B %d, %Y'),
                'Text': c.short_text,
                'Likes': c.like_cnt,
            }
            for c in comments[0:5]
        ]).sort_values('Likes', ascending=False)

        st.table(like_comments)

    #
    # Summaries/themes and suggestions
    #

    display_summary(db_session, video_id)

    display_suggest(db_session, video_id)

    #
    # Disclaimer thing
    #

    st.markdown('---')
    st.markdown('**SOURCING**: Comments scraped from YouTube, themes and '
                'suggestions identified with GPT-4.')


def channel_card(db_session, video_stats):
    as_of_date = get_as_of_date() \
        .replace(tzinfo=ZoneInfo('America/New_York')) \
        .strftime('%I:%M:%S %p, %B %d, %Y ET')

    video_stats = video_stats.copy()
    n_videos = video_stats.shape[0]
    total_comments = video_stats['comment_cnt'].sum()
    total_views = video_stats['view_cnt'].sum()

    avg_sentiment = (video_stats['negative'] * video_stats['comment_cnt']).sum()
    avg_sentiment = avg_sentiment / video_stats['comment_cnt'].sum()
    avg_sentiment = 1 + (-1 * avg_sentiment)

    video_stats.rename({'negative': 'Sentiment'}, axis=1, inplace=True)

    #
    # Top card
    #

    col1, _, _, col4 = st.columns(4)
    with col1:
        st.image('assets/frontline.png')
    with col4:
        st.image('assets/mit.png')

    st.markdown('---')

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image('assets/frontline-circle.png', width=200)
        st.image('assets/youtube.png', width=200)
    with col2:
        st.markdown('**FRONTLINE CHANNEL REPORT**')

        st.write('**Dataset**: All Documentaries')
        st.write(f'**Data As-of Date:** {as_of_date}')
        st.write(f'**Total Views:** {total_views:,} (avg {int(total_views/n_videos):,} / video)')
        st.write(f'**Total Comments:** {total_comments:,} (avg {int(total_comments/n_videos):,} / video)')
        st.write(f'**Avg Sentiment:** {avg_sentiment:.2f}')

    st.markdown('<hr style="height:4px;border:none;color:#ff0000;' +
                'background-color:#ff0000;" />', unsafe_allow_html=True)

    display_summary(db_session, None)

    display_topics(db_session)

    display_video_change_alerts()

    display_sentiment_chart(db_session, video_stats)

    display_superfans(db_session)

    display_suggest(db_session, None)

    st.markdown('---')
    st.markdown('**SOURCING**: Comments scraped from YouTube, themes and '
                'suggestions identified with GPT-4. Detailed topics are based '
                'on clustering large numbers of comments in order to be '
                'comprehensive and may not line up exactly with themes.')


def run():
    video_stats = get_video_stats()
    db_factory = get_db(DB_PATH)

    display_tooltip_style()

    with db_factory() as db_session:
        ctab, vtab = st.tabs(['Channel', 'Video'])

        with ctab:
            channel_card(db_session, video_stats)

        with vtab:
            video_card(db_session, video_stats)
