"""Interactive figures over validated service outputs; no model or impact inference."""
from __future__ import annotations
import plotly.graph_objects as go

COLORS = {'GP01': '#18684d', 'GP02': '#527797', 'GP03': '#ad7b20', 'GP04': '#76934e'}
NAMES = {'GP01': 'EV finance', 'GP02': 'Green home', 'GP03': 'Solar finance', 'GP04': 'Paperless'}
CONFIG = {'displayModeBar': False, 'displaylogo': False, 'responsive': True}


def polish(fig, *, height=310):
    fig.update_layout(height=height, margin=dict(l=10, r=15, t=35, b=45),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Atkinson, Segoe UI, sans-serif', color='#24483a', size=12),
        legend=dict(orientation='h', y=1.15, x=0), hoverlabel=dict(bgcolor='#fff', font_size=13),
        dragmode=False)
    fig.update_xaxes(gridcolor='#e4ece5', zeroline=False, fixedrange=True)
    fig.update_yaxes(gridcolor='#e4ece5', zeroline=False, fixedrange=True)
    return fig


def products_chart(products):
    labels = [NAMES[p['product_code']] for p in products]
    fig = go.Figure()
    for key, label, color in [('structurally_eligible', 'Relevant cohort', '#acc6b3'),
                             ('displayed_opportunities', 'Shown now', '#246b50'),
                             ('high_priority', 'High priority', '#92b869')]:
        fig.add_bar(y=labels, x=[p[key] for p in products], name=label, orientation='h',
                    marker_color=color, marker_line=dict(color='#476e55', width=1), hovertemplate='%{y}<br>'+label+': %{x} customers<extra></extra>')
    fig.update_layout(barmode='group', bargap=.3)
    fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(title='Customers')
    # Room for product names: automargin measures before the web font loads.
    return polish(fig).update_layout(margin_l=105)


def regions_chart(segments):
    fig = go.Figure()
    if segments:
        maximum = max(s['customers'] for s in segments)
        fig.add_scatter(x=[NAMES[s['product_code']] for s in segments],
            y=[s['region'] for s in segments], mode='markers',
            marker=dict(size=[s['customers'] for s in segments], sizemode='area',
                        sizeref=2*maximum/(35**2), sizemin=6,
                        color=[COLORS[s['product_code']] for s in segments], opacity=.85,
                        line=dict(width=1.5, color='#fff')),
            customdata=[[s['region'], s['product_code'], s['customers'], s['mean_propensity'], s['mean_confidence']] for s in segments],
            hovertemplate='%{customdata[0]} · %{x}<br>%{customdata[2]} opportunities<br>Mean likelihood %{customdata[3]:.1%}<br>Mean confidence %{customdata[4]:.1%}<extra></extra>')
    fig.update_xaxes(showgrid=False, categoryorder='array', categoryarray=list(NAMES.values()))
    fig.update_yaxes(showgrid=True)
    return polish(fig, height=max(290, len({s['region'] for s in segments})*27+70))


def trend_chart(history, metric='customers'):
    fig = go.Figure()
    dates = [s['date'] for s in history]
    if metric == 'products':
        for code in NAMES:
            fig.add_scatter(x=dates, y=[s.get('opportunities_by_product',{}).get(code) for s in history],
                name=NAMES[code], mode='lines+markers', line=dict(color=COLORS[code], width=2.5),
                marker=dict(size=7), hovertemplate='%{x|%d %b %Y}<br>%{y} opportunities<extra>%{fullData.name}</extra>')
        fig.update_yaxes(title='Opportunities')
    else:
        fig.add_scatter(x=dates, y=[s['customers_with_opportunities'] for s in history],
            mode='lines+markers', line=dict(color='#246b50',width=3), marker=dict(size=8),
            fill='tozeroy', fillcolor='rgba(146,184,105,.12)',
            hovertemplate='%{x|%d %b %Y}<br>%{y} customers<extra></extra>')
        fig.update_yaxes(title='Customers with opportunities', rangemode='tozero')
    return polish(fig, height=280)


def likelihood_chart(propensity, cohort_mean):
    fig = go.Figure(go.Indicator(mode='number+gauge', value=100*propensity,
        number=dict(suffix='%', valueformat='.1f', font=dict(size=26,color='#24483a')),
        domain=dict(x=[0,1], y=[0,1]),
        gauge=dict(shape='bullet', axis=dict(range=[0,100],tickvals=[0,50,100],ticksuffix='%'),
                   bar=dict(color='#246b50',thickness=.6), bgcolor='#edf3e9', borderwidth=0,
                   threshold=dict(line=dict(color='#a67521',width=3),thickness=.85,value=100*cohort_mean) if cohort_mean is not None else {})))
    fig.update_layout(height=95,margin=dict(l=0,r=15,t=8,b=22),
                      paper_bgcolor='rgba(0,0,0,0)',font=dict(family='Atkinson, Segoe UI, sans-serif',color='#24483a'))
    return fig
