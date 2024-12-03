# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
# add parent directory to path
import os
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import numpy as np
import plotly.offline as offline
import utils.utils as utils
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import time
import pandas as pd
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from joblib import Parallel, delayed
from matplotlib.colorbar import ColorbarBase


def plot_contour_sdf(x_grid, y_grid, z_grid, clean_points,
                           example_idx=0, n_gt=None, n_pred=None,
                           show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                           nonmnfld_points=None, marker_size=10,
                           plot_second_derivs=False, mnfld_segments=None):
    print('plotting started'); t0 = time.time()
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()[:500]
        if mnfld_segments is not None:
            mnfld_segments = mnfld_segments[example_idx].detach().cpu().numpy()[:500]
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()[:500]
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()[:500]

    traces = []

    # # import pdb; pdb.set_trace()
    # import matplotlib
    # colors = dict(matplotlib.colors.cnames.items())
    # assert mnfld_segments.max() < len(colors)
    # mnfld_segment_colors = np.array(sorted(list(colors.values())))[mnfld_segments]

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        # import pdb; pdb.set_trace()
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                    # mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
                                    mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)' if mnfld_segments is None else mnfld_segments))
                                    # mode='markers', marker=dict(size=marker_size, color=mnfld_segment_colors))
        traces.append(clean_points_scatter) # clean points scatter

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter


    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    # dist_img = utils.plotly_fig2array(fig1)
    # return dist_img
    return fig1

def plot_contours(x, y, z, colorscale='Geyser', fig=None):
    # plots a contour image given hight map matrix (2d array)
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], ),
                       yaxis=dict(side="left", range=[-1, 1], ),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='distance map', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Serif', size=24, color='red'))
                       )
    if fig is None:
        traces = []
    else:
        traces = list(fig.data)
    traces.append(go.Contour(x=x.flatten(), y=y.flatten(), z=z.flatten(),
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ),
                               ))  # contour trace

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_scatter(x, y):
    layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                           yaxis=dict(range=[-1, 1], autorange=False),
                                                           aspectratio=dict(x=1, y=1)),
                                                           title=dict(text='scatter plot'))
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=16
        )
    ), layout=layout)

    fig.show()


def plot_contour_and_scatter_sal_training(clean_points, decoder, latent,
                                          example_idx=0, n_gt=None, n_pred=None,
                                          show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                                          res=256):
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation

    clean_points = clean_points[example_idx].detach().cpu().numpy()

    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()
    x, y, grid_points = utils.get_2d_grid_uniform(resolution=res, range=1.2, device=latent.device)
    grid_points = torch.cat([latent[example_idx].expand(grid_points.shape[0], -1), grid_points], dim=1)
    z = decoder(grid_points).detach().cpu().numpy()

    traces = []
    # plot implicit function contour
    traces.append(go.Contour(x=x, y=y, z=z.reshape(x.shape[0], x.shape[0]),
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    # plot clean points and noisy points scatters
    traces.append(go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                mode='markers', marker=dict(size=16, color=(0, 0, 0)))) # clean points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    img = utils.plotly_fig2array(fig1)
    return img


# def plot_contour_scatter_and_curl(clean_points, decoder, latent,
#                                           example_idx=0, n_gt=None, n_pred=None, z_gt=None,
#                                           show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
#                                           res=128, nonmnfld_points=None):
#     # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
#
#     clean_points = clean_points[example_idx].detach().cpu().numpy()
#
#     if n_gt is not None:
#         n_gt = n_gt[example_idx].detach().cpu().numpy()
#     if n_pred is not None:
#         n_pred = n_pred[example_idx].detach().cpu().numpy()
#     x, y, grid_points = utils.get_2d_grid_uniform(resolution=res, range=1.2, device=latent.device)
#     grid_points.requires_grad_()
#     grid_points_latent = torch.cat([latent[example_idx].expand(grid_points.shape[0], -1), grid_points], dim=1)
#     z = decoder(grid_points_latent)
#     z_np = z.detach().cpu().numpy()
#     traces = []
#     # plot implicit function contour
#
#     traces.append(go.Contour(x=x, y=y, z=z_np.reshape(x.shape[0], x.shape[0]),
#                                colorscale=colorscale,
#                                # autocontour=True,
#                                contours=dict(
#                                    start=-1,
#                                    end=1,
#                                    size=0.025,
#                                ), showscale=show_bar
#                                ))  # contour trace
#
#     # plot clean points and noisy points scatters
#     clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
#                                 mode='markers', marker=dict(size=16, color=(0, 0, 0)))
#     traces.append(clean_points_scatter) # clean points scatter
#
#     if nonmnfld_points is not None:
#         nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
#         nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
#                                           mode='markers', marker=dict(size=16, color='rgb(255, 255, 255)'))
#         traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter
#
#
#     # plot normal vectors:
#     if n_gt is not None:
#         f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
#         traces.append(f.data[0])
#     if n_pred is not None:
#         f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
#                              line=dict(color='rgb(255, 0, 0)'))
#         traces.append(f.data[0])
#
#     layout = go.Layout(width=1200, height=1200,
#                        xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
#                        yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
#                        scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
#                                   yaxis=dict(range=[-1, 1], autorange=False),
#                                   aspectratio=dict(x=1, y=1)),
#                        showlegend=False,
#                        title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
#                                   font=dict(family='Sherif', size=24, color='red'))
#                        )
#
#     fig1 = go.Figure(data=traces, layout=layout)
#     offline.plot(fig1, auto_open=False)
#     dist_img = utils.plotly_fig2array(fig1)
#
#     # plot curl
#     grid_grad = utils.gradient(grid_points, z)
#     dx = utils.gradient(grid_points, grid_grad[:, 0], create_graph=False, retain_graph=True)
#     dy = utils.gradient(grid_points, grid_grad[:, 1], create_graph=False, retain_graph=False)
#     grid_curl = (dx[:, 1] - dy[:, 0]).cpu().detach().numpy()
#
#     traces = [clean_points_scatter]
#     traces.append(go.Contour(x=x, y=y, z=grid_curl.reshape(x.shape[0], x.shape[0]),
#                                colorscale=colorscale,
#                                # autocontour=True,
#                                contours=dict(
#                                    start=-1e-5,
#                                    end=1e-5,
#                                    size=0.25*1e-5,
#                                ), showscale=show_bar
#                                ))  # contour trace
#     # f = ff.create_quiver(grid_points[:, 0].detach().cpu().numpy(), grid_points[:, 1].detach().cpu().numpy(),
#     # grid_grad[:, 0].detach().cpu().numpy(), grid_grad[:, 1].detach().cpu().numpy(),
#     #                      line=dict(color='rgb(0, 0, 0)'))
#     # traces.append(f.data[0])
#     fig2 = go.Figure(data=traces, layout=layout)
#     offline.plot(fig2, auto_open=False)
#     curl_img = utils.plotly_fig2array(fig2)
#
#     # plot eikonal
#     traces = [clean_points_scatter]
#     eikonal_term = ((grid_grad.norm(2, dim=-1) - 1) ** 2).cpu().detach().numpy()
#     traces.append(go.Contour(x=x, y=y, z=eikonal_term.reshape(x.shape[0], x.shape[0]),
#                                colorscale=colorscale,
#                                # autocontour=True,
#                                contours=dict(
#                                    start=-1,
#                                    end=1,
#                                    size=0.025,
#                                ), showscale=show_bar
#                                ))  # contour trace
#     fig3 = go.Figure(data=traces, layout=layout)
#     offline.plot(fig3, auto_open=False)
#     eikonal_img = utils.plotly_fig2array(fig3)
#
#     #plot divergence
#     grid_div = (dx[:, 0] + dy[:, 1]).detach().cpu().numpy()
#     traces = [clean_points_scatter]
#     traces.append(go.Contour(x=x, y=y, z=grid_div.reshape(x.shape[0], x.shape[0]),
#                                colorscale=colorscale,
#                                # autocontour=True,
#                                contours=dict(
#                                    start=-10,
#                                    end=10,
#                                    size=0.25,
#                                ), showscale=show_bar
#                                ))  # contour trace
#     fig4 = go.Figure(data=traces, layout=layout)
#     offline.plot(fig2, auto_open=False)
#     div_img = utils.plotly_fig2array(fig4)
#
#     #plot z difference image
#     z_diff = np.abs(np.abs(np.reshape(z_np, [res, res])) - np.abs(z_gt))
#     traces = [clean_points_scatter]
#     traces.append(go.Contour(x=x, y=y, z=z_diff.reshape(x.shape[0], x.shape[0]),
#                                colorscale=colorscale,
#                                # autocontour=True,
#                                contours=dict(
#                                    start=-0.5,
#                                    end=0.5,
#                                    size=0.025,
#                                ), showscale=show_bar
#                                ))  # contour trace
#     fig5 = go.Figure(data=traces, layout=layout)
#     offline.plot(fig5, auto_open=False)
#     z_diff_img = utils.plotly_fig2array(fig5)
#
#     return dist_img, curl_img, eikonal_img, div_img, z_diff_img, grid_div

def plot_contour_props(x_grid, y_grid, z_grid, clean_points,
                           z_diff,
                           example_idx=0, n_gt=None, n_pred=None,
                           show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                           nonmnfld_points=None, marker_size=10,
                           plot_second_derivs=False):
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []
    print(z_grid.min(), z_grid.max())

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        num = 200
        clean_points_scatter = go.Scatter(x=clean_points[:num, 0], y=clean_points[:num, 1],
                                    mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter) # clean points scatter

    if nonmnfld_points is not None:
        num = 200
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:num, 0], y=nonmnfld_points[:num, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter


    # plot normal vectors:
    if n_gt is not None:
        num = 20
        f = ff.create_quiver(clean_points[:num, 0], clean_points[:num, 1], n_gt[:num, 0], n_gt[:num, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        num = 20
        f = ff.create_quiver(clean_points[:num, 0], clean_points[:num, 1], n_pred[:num, 0], n_pred[:num, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image")
    return dist_img


def plot_segmentation_and_centroids(points, segments, centroids, centroid_segment_idx, example_idx=0, marker_size=10, show_ax=True, title_text='', cmax=24):
    traces = []
    n_centroids = len(centroids)
    n_segments = len(centroid_segment_idx)
    if n_segments > 48:
        raise
    elif n_segments > cmax:
        cmax = n_segments
        color_discrete_sequence = np.concatenate([px.colors.qualitative.Alphabet, px.colors.qualitative.Dark24])
    else:
        # color_discrete_sequence = px.colors.qualitative.Alphabet
        color_discrete_sequence = px.colors.qualitative.Dark24
    points = points[example_idx].detach().cpu().numpy()
    segments = segments[example_idx].detach().cpu().numpy()



    # plot clean points and noisy points scatters
    clean_points_scatter = go.Scatter(x=points[:, 0], y=points[:, 1],  mode='markers',
                                      marker=dict(size=marker_size, color=segments,
                                                  colorscale=color_discrete_sequence,
                                                  cmin=0, cmax=cmax))
    traces.append(clean_points_scatter)

    # plot the centroids larger and in different marker shape
    centroids_scatter = go.Scatter(x=centroids[:, 0], y=centroids[:, 1],  mode='markers',
                                   marker=dict(size=marker_size*2, color=np.arange(0, n_centroids),
                                               colorscale=color_discrete_sequence,
                                               symbol='star', cmin=0, cmax=cmax))
    traces.append(centroids_scatter)

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    offline.plot(fig, auto_open=False)
    segments_img = utils.plotly_fig2array(fig)

    # # Create a scatter plot
    # plt.figure(figsize=(8, 8))
    # plt.scatter(points[:, 0], points[:, 1], c=segments, cmap='tab20', alpha=1.0, s=marker_size, vmin=0, vmax=20)
    # plt.gca().set_prop_cycle(None)
    # plt.scatter(centroids[:, 0], centroids[:, 1], c=centroid_segment_idx, cmap='tab20', alpha=1.0, marker='*', s=marker_size*2, vmin=0, vmax=20)
    # plt.xlim([-1, 1])
    # plt.ylim([-1, 1])
    # # Customize the plot
    # plt.title('2D Point Scatter with Color Segments')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.colorbar(label='Segments')
    #
    # # Save the plot to an array
    # fig = plt.gcf()
    # fig.canvas.draw()
    # data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # segments_img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #
    # # Close the plot to release memory
    # plt.close()
    print("Finished plotting segments image")
    return segments_img

def plot_contour_div_props(x_grid, y_grid, z_grid, clean_points,
                           z_diff, eikonal_term, grid_div, grid_curl, var_regmoesdf, z_all_experts,
                           example_idx=0, n_gt=None, n_pred=None, n_pred_experts=None,
                           show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                           nonmnfld_points=None, marker_size=10,
                           plot_second_derivs=False):
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()
    if n_pred_experts is not None:
        n_pred_experts = n_pred_experts[example_idx].detach().cpu().numpy()

    traces = []

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                    mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter) # clean points scatter

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter


    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image")

    # plot expert sdfs
    if z_all_experts is not None:
        experts_dist_img = []
        for i, z_grid_expert in enumerate(z_all_experts):
            if clean_points is not None:
                traces = [clean_points_scatter]

                # plot implicit function contour
            traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid_expert,
                                     colorscale=colorscale,
                                     # autocontour=True,
                                     contours=dict(
                                         start=-1,
                                         end=1,
                                         size=0.025,
                                     ), showscale=show_bar
                                     ))  # contour trace
            traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid_expert,
                                     contours=dict(start=0, end=0, coloring='lines'),
                                     line=dict(width=3),
                                     showscale=False,
                                     colorscale=[[0, 'rgb(100, 100, 100)'],
                                                 [1, 'rgb(100, 100, 100)']]))  # black bold zero line

            # plot normal vectors:
            if n_gt is not None:
                f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1],
                                     line=dict(color='rgb(0, 255, 0)'))
                traces.append(f.data[0])
            if n_pred_experts is not None:
                f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1],
                                     n_pred_experts[i, :, 0], n_pred_experts[i, :, 1],
                                     line=dict(color='rgb(255, 0, 0)'))
                traces.append(f.data[0])

            layout = go.Layout(width=800, height=800,
                               xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax,
                                          visible=show_ax),
                               yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                               scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                          yaxis=dict(range=[-1, 1], autorange=False),
                                          aspectratio=dict(x=1, y=1)),
                               showlegend=False,
                               title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                          font=dict(family='Sherif', size=24, color='red'))
                               )

            fig1 = go.Figure(data=traces, layout=layout)
            offline.plot(fig1, auto_open=False)
            experts_dist_img.append(utils.plotly_fig2array(fig1))
            print("Finished computing expert sign distance image")

    # plot eikonal
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=eikonal_term,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    fig3 = go.Figure(data=traces, layout=layout)
    offline.plot(fig3, auto_open=False)
    eikonal_img = utils.plotly_fig2array(fig3)
    print("Finished computing eikonal image")
    #plot z difference image
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_diff,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-0.5,
                                   end=0.5,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    fig5 = go.Figure(data=traces, layout=layout)
    offline.plot(fig5, auto_open=False)
    z_diff_img = utils.plotly_fig2array(fig5)
    print("Finished computing gt distance image")
    if plot_second_derivs:
        #plot divergence
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_div, -10, 10),
                                   colorscale=colorscale,
                                   # autocontour=True,
                                   contours=dict(
                                       start=-10,
                                       end=10,
                                       size=1,
                                   ),
                                   showscale=show_bar
                                   ))  # contour trace
        fig4 = go.Figure(data=traces, layout=layout)
        offline.plot(fig4, auto_open=False)
        div_img = utils.plotly_fig2array(fig4)
        print("Finished divergence image")
        # plot curl
        # if clean_points is not None:
        #     traces = [clean_points_scatter]
        # traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_curl, -1e-4, 1e-4),
        #                            colorscale=colorscale,
        #                            # autocontour=True,
        #                            contours=dict(
        #                                start=-1e-4,
        #                                end=1e-4,
        #                                size=1e-5,
        #                            ),
        #                             showscale=show_bar
        #                            ))  # contour trace

        # fig2 = go.Figure(data=traces, layout=layout)
        # offline.plot(fig2, auto_open=False)
        # curl_img = utils.plotly_fig2array(fig2)
        # print("Finished computing curl image")
        curl_img = np.zeros_like(dist_img)
    else:
        # div_img, z_diff_img, curl_img = np.zeros_like(dist_img), np.zeros_like(dist_img), np.zeros_like(dist_img)
        div_img, curl_img = None, None
    print("Finished computing eikonal image")

    #plot moe sdf regularizer image
    if var_regmoesdf is not None:
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=var_regmoesdf,
                                   colorscale=colorscale,
                                   # autocontour=True,
                                   contours=dict(
                                       start=0,
                                       end=0.1,
                                       size=0.005,
                                   ), showscale=show_bar
                                   ))  # contour trace
        fig6 = go.Figure(data=traces, layout=layout)
        offline.plot(fig6, auto_open=False)
        regmoesdf_img = utils.plotly_fig2array(fig6)
        print("Finished computing sdf moe regularizer image")
    else:
        regmoesdf_img = None
        experts_dist_img = None

    return dist_img, curl_img, eikonal_img, div_img, z_diff_img, regmoesdf_img, experts_dist_img

def plot_experts(points, expert_idx, q, q_grad, colormap='tab10', example_idx=0, n_cols=4, clim=(0.0, 0.5),
                 show_axis=False):


    points = points[example_idx].detach().cpu().numpy()
    new_range = (-1, 1)
    n_experts = q.shape[0]
    n_rows = np.ceil(n_experts/n_cols).astype(int)

    ################################### plot q heatmaps in a simgle image ###################################
    fig = plt.figure(1,(16, 16))
    canvas = fig.canvas
    axs = []
    for expert_id in range(n_experts):
        q_expert = q[expert_id]
        ax = fig.add_subplot(n_rows, n_cols, expert_id + 1)
        axs.append(ax)
        ax.set_title('Expert {} q'.format(expert_id))
        im = ax.imshow(np.flipud(q_expert), cmap='jet', interpolation='none',
                       extent=(new_range[0], new_range[1], new_range[0], new_range[1]), clim=clim)
        # Plot the scatter points on top of the image
        ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], c='black', marker='o', s=20)

        # Set axis limits and labels
        plt.axis('off')
        ax.set_xlim(new_range[0], new_range[1])
        ax.set_ylim(new_range[0], new_range[1])
        ax.set_xticks(np.arange(new_range[0], new_range[1] + 0.5, 0.5))
        ax.set_yticks(np.arange(new_range[0], new_range[1] + 0.5, 0.5))
        # ax.set_aspect('equal')  # Ensure equal aspect ratio

    # Create a colorbar with discrete values
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    # plt.show()
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    q_image_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    ################################### plot q heatmaps in separate images ###################################
    q_image_array_list = []
    for expert_id in range(n_experts):
        q_expert = q[expert_id]
        fig = plt.figure(1, (16, 16))
        canvas = fig.canvas
        ax = fig.add_subplot(111)
        ax.imshow(np.flipud(q_expert), cmap='jet', interpolation='none',
                  extent=(new_range[0], new_range[1], new_range[0], new_range[1]), clim=clim)
        ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], c='black', marker='o', s=200)
        # Set axis limits and labels
        plt.axis('off')
        ax.set_xlim(new_range[0], new_range[1])
        ax.set_ylim(new_range[0], new_range[1])
        ax.set_xticks(np.arange(new_range[0], new_range[1] + 0.5, 0.5))
        ax.set_yticks(np.arange(new_range[0], new_range[1] + 0.5, 0.5))
        # ax.set_aspect('equal')  # Ensure equal aspect ratio
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        q_image_array_list.append(image_array.reshape(canvas.get_width_height()[::-1] + (3,)))
        plt.close(fig)

    #################################### plot expert idx ####################################
    # Create a figure
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    # Create a custom colormap based on the unique values in expert_idx
    num_colors = n_experts
    custom_colormap = plt.get_cmap(colormap, num_colors)
    custom_norm = BoundaryNorm(np.arange(-0.5, num_colors, 1), custom_colormap.N)

    # Plot the image with the custom colormap
    im = ax.imshow(np.flipud(expert_idx), cmap=custom_colormap, norm=custom_norm, interpolation='none',
                   extent=(new_range[0], new_range[1], new_range[0], new_range[1]))

    # Plot the scatter points on top of the image
    ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], c='black', marker='o', s=20)

    # Create a colorbar with discrete values
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(-0.5, num_colors, 1))
    # Set the tick labels to match the unique values in expert_idx
    values = range(n_experts)
    positions = [v for v in values]  # Centering the ticks
    cbar.set_ticks(positions)
    cbar.set_ticklabels(values)
    cbar.set_label('Expert ID')
    # plt.show()
    # Set axis limits and labels
    ax.set_xlim(new_range[0], new_range[1])
    ax.set_ylim(new_range[0], new_range[1])
    if show_axis:
        ax.set_xticks(np.arange(new_range[0], new_range[1]+0.5, 0.5))
        ax.set_yticks(np.arange(new_range[0], new_range[1]+0.5, 0.5))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    # ax.set_aspect('equal')  # Ensure equal aspect ratio
    fig.tight_layout()
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    ############################# plot q gradient heatmaps #############################
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.flipud(q_grad), cmap='jet', interpolation='none',
                   extent=(new_range[0], new_range[1], new_range[0], new_range[1]), clim=[0, 10])
    ax.scatter(np.array(points)[:, 0], np.array(points)[:, 1], c='black', marker='o', s=20)
    ax.set_xlim(new_range[0], new_range[1])
    ax.set_ylim(new_range[0], new_range[1])
    ax.set_xticks(np.arange(new_range[0], new_range[1]+0.5, 0.5))
    ax.set_yticks(np.arange(new_range[0], new_range[1]+0.5, 0.5))
    # ax.set_aspect('equal')  # Ensure equal aspect ratio
    # Create a colorbar with discrete values
    cbar = plt.colorbar(im, ax=ax)
    # plt.show()
    canvas.draw()
    q_grad_image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    q_grad_image_array = q_grad_image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    plt.close()

    return image_array, q_image_array, q_grad_image_array, q_image_array_list



def plot_init_contour_div_props(x_grid, y_grid, z_grid, clean_points,
                           z_diff, eikonal_term, grid_div, grid_curl,
                           example_idx=0, n_gt=None, n_pred=None,
                           show_ax=False, show_bar=True, title_text='', colorscale='Geyser',
                           nonmnfld_points=None, marker_size=10,
                           plot_second_derivs=False):
    import time
    t0 = time.time()
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                            colorbar=dict(tickfont=dict(size=25)),
                            # contours_coloring='heatmap' # For smoothing
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                #    size=0.05, # Siren
                                #    size=0.1, # geometric sine
                                   size=0.1, # mfgi
                               ), showscale=show_bar
                               ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                    mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter) # clean points scatter

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter


    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image, {:.2f}s".format(time.time()-t0))
    t0 = time.time()
    # plot eikonal
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=eikonal_term,
                            colorbar=dict(tickfont=dict(size=25)),
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.1,
                               ), showscale=show_bar
                               ))  # contour trace
    fig3 = go.Figure(data=traces, layout=layout)
    offline.plot(fig3, auto_open=False)
    eikonal_img = utils.plotly_fig2array(fig3)
    print("Finished computing eikonal image, {:.2f}s".format(time.time()-t0))
    t0 = time.time()
    #plot z difference image
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_diff,
                            colorbar=dict(tickfont=dict(size=25)),
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-0.5,
                                   end=0.5,
                                   size=0.05,
                               ), showscale=show_bar
                               ))  # contour trace
    fig5 = go.Figure(data=traces, layout=layout)
    offline.plot(fig5, auto_open=False)
    z_diff_img = utils.plotly_fig2array(fig5)
    print("Finished computing gt distance image, {:.2f}s".format(time.time()-t0))
    t0 = time.time()
    if plot_second_derivs:
        #plot divergence
        traces = []
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_div, -50, 50),
                                colorbar=dict(tickfont=dict(size=25)),
                                   colorscale=colorscale,
                                   # autocontour=True,
                                   contours=dict(
                                       start=-50,
                                       end=50,
                                       size=1,
                                   ),
                                   contours_coloring='heatmap',
                                #    showscale=show_bar
                                   ))  # contour trace
        fig4 = go.Figure(data=traces, layout=layout)
        offline.plot(fig4, auto_open=False)
        div_img = utils.plotly_fig2array(fig4)
        print("Finished divergence image, {:.2f}s".format(time.time()-t0))
        t0 = time.time()
        # # plot curl
        # traces = []
        # if clean_points is not None:
        #     traces = [clean_points_scatter]
        # traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_curl, -1e-4, 1e-4),
        #                         colorbar=dict(tickfont=dict(size=25)),
        #                            colorscale=colorscale,
        #                            # autocontour=True,
        #                            contours=dict(
        #                                start=-1e-4,
        #                                end=1e-4,
        #                                size=1e-5,
        #                            ),
        #                             showscale=show_bar
        #                            ))  # contour trace

        # fig2 = go.Figure(data=traces, layout=layout)
        # offline.plot(fig2, auto_open=False)
        # curl_img = utils.plotly_fig2array(fig2)
        # print("Finished computing curl image, {:.2f}s".format(time.time()-t0))
        # t0 = time.time()
        curl_img = None
    else:
        div_img, z_diff_img, curl_img = np.zeros_like(dist_img), np.zeros_like(dist_img), np.zeros_like(dist_img)
    return dist_img, curl_img, eikonal_img, div_img, z_diff_img


def plot_gt_contour_div_props(x_grid, y_grid, z_grid, clean_points,
                           z_diff, eikonal_term, grid_div, grid_curl,
                           example_idx=0, n_gt=None, n_pred=None,
                           show_ax=True, show_bar=True, title_text='', colorscale='Geyser',
                           nonmnfld_points=None, marker_size=10,
                           plot_second_derivs=False):
    import time
    t0 = time.time()
    # plot contour and scatter plot given input points, SAL decoder and the point's laten representation
    div_img, curl_img = 0, 0
    if clean_points is not None:
        clean_points = clean_points[example_idx].detach().cpu().numpy()
    if n_gt is not None:
        n_gt = n_gt[example_idx].detach().cpu().numpy()
    if n_pred is not None:
        n_pred = n_pred[example_idx].detach().cpu().numpy()

    traces = []

    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-1,
                                   end=1,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=3),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # black bold zero line

    # plot clean points and noisy points scatters
    if clean_points is not None:
        clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                    mode='markers', marker=dict(size=marker_size, color='rgb(0, 0, 0)'))
        traces.append(clean_points_scatter) # clean points scatter

    if nonmnfld_points is not None:
        nonmnfld_points = nonmnfld_points[example_idx].detach().cpu().numpy()
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                          mode='markers', marker=dict(size=marker_size, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # nonmnfld points scatter


    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])
    if n_pred is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_pred[:, 0], n_pred[:, 1],
                             line=dict(color='rgb(255, 0, 0)'))
        traces.append(f.data[0])

    layout = go.Layout(width=800, height=800,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig1 = go.Figure(data=traces, layout=layout)
    offline.plot(fig1, auto_open=False)
    dist_img = utils.plotly_fig2array(fig1)
    print("Finished computing sign distance image, {:.2f}s".format(time.time()-t0))
    t0 = time.time()
    # plot eikonal
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=eikonal_term,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=0,
                                   end=2,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    fig3 = go.Figure(data=traces, layout=layout)
    offline.plot(fig3, auto_open=False)
    eikonal_img = utils.plotly_fig2array(fig3)
    print("Finished computing eikonal image, {:.2f}s".format(time.time()-t0))
    t0 = time.time()
    #plot z difference image
    traces = []
    if clean_points is not None:
        traces = [clean_points_scatter]
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_diff,
                               colorscale=colorscale,
                               # autocontour=True,
                               contours=dict(
                                   start=-0.5,
                                   end=0.5,
                                   size=0.025,
                               ), showscale=show_bar
                               ))  # contour trace
    fig5 = go.Figure(data=traces, layout=layout)
    offline.plot(fig5, auto_open=False)
    z_diff_img = utils.plotly_fig2array(fig5)
    print("Finished computing gt distance image, {:.2f}s".format(time.time()-t0))
    t0 = time.time()
    if plot_second_derivs:
        #plot divergence
        traces = []
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_div, -50, 50),
                                   colorscale=colorscale,
                                   # autocontour=True,
                                   contours=dict(
                                       start=-50,
                                       end=50,
                                       size=1,
                                   ),
                                   contours_coloring='heatmap',
                                #    showscale=show_bar
                                   ))  # contour trace
        fig4 = go.Figure(data=traces, layout=layout)
        offline.plot(fig4, auto_open=False)
        div_img = utils.plotly_fig2array(fig4)
        print("Finished divergence image, {:.2f}s".format(time.time()-t0))
        t0 = time.time()
        # plot curl
        traces = []
        if clean_points is not None:
            traces = [clean_points_scatter]
        traces.append(go.Contour(x=x_grid, y=y_grid, z=np.clip(grid_curl, -1e-4, 1e-4),
                                   colorscale=colorscale,
                                   # autocontour=True,
                                   contours=dict(
                                       start=-1e-4,
                                       end=1e-4,
                                       size=1e-5,
                                   ),
                                    showscale=show_bar
                                   ))  # contour trace

        fig2 = go.Figure(data=traces, layout=layout)
        offline.plot(fig2, auto_open=False)
        curl_img = utils.plotly_fig2array(fig2)
        print("Finished computing curl image, {:.2f}s".format(time.time()-t0))
        t0 = time.time()
        # curl_img = None
    else:
        div_img, z_diff_img, curl_img = np.zeros_like(dist_img), np.zeros_like(dist_img), np.zeros_like(dist_img)
    return dist_img, curl_img, eikonal_img, div_img, z_diff_img


def plot_paper_teaser_images(x_grid, y_grid, z_grid, clean_points, grid_normals, colorscale='Geyser', show_ax=False, output_path='/mnt/3.5TB_WD/PycharmProjects/DiBS/sanitychecks/vis/'):
    os.makedirs(output_path, exist_ok=True)
    layout1 = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    layout2 = go.Layout(width=1200, height=1200,  plot_bgcolor='rgb(255, 255, 255)',
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=True, zeroline=show_ax, visible=True, gridcolor='LightGrey'),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=True, zeroline=show_ax, visible=True, gridcolor='LightGrey'),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    traces2 = []
    traces = []
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                             contours=dict(start=-3, end=3, size=0.1), showscale=False))  # contour trace
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5, dash="dash"),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace (zls)

    # plot clean points and noisy points scatters
    scatter_points = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                mode='markers', marker=dict(size=40, color='rgb(50, 50, 50)'))
    traces.append(scatter_points)
    traces2.append(scatter_points)
    traces.append(ff.create_quiver(x_grid, y_grid, grid_normals[:, 0], grid_normals[:, 1], line=dict(color='rgb(0, 0, 255)'),
                         scale=.04).data[0])



    fig1 = go.Figure(data=traces, layout=layout1)
    fig2 = go.Figure(data=traces2, layout=layout2)
    # fig2.show()
    offline.plot(fig1, auto_open=False)
    offline.plot(fig2, auto_open=False)
    img1 = utils.plotly_fig2array(fig1)
    img2 = utils.plotly_fig2array(fig2)
    im = Image.fromarray(img1)
    im.save(os.path.join(output_path, "teaser1.png"))
    im = Image.fromarray(img2)
    im.save(os.path.join(output_path, "teaser2.png"))


def plot_shape_data(x_grid, y_grid, z_grid, clean_points, n_gt=None, show_ax=True, show_bar=True,
                    title_text='', colorscale='Geyser', nonmnfld_points=None, divergence=None, grid_normals=None):
    # plot contour and scatter plot given input points

    traces = []
    # plot implicit function contour
    if divergence is None:
        traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                                   contours=dict( start=-1, end=1, size=0.025), showscale=show_bar ))  # contour trace
    else:
        traces.append(go.Contour(x=x_grid, y=y_grid, z=divergence, colorscale=colorscale,
                                   contours=dict( start=-3, end=3, size=0.25), showscale=show_bar ))  # contour trace

    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace

    # plot clean points and noisy points scatters
    clean_points_scatter = go.Scatter(x=clean_points[:, 0], y=clean_points[:, 1],
                                mode='markers', marker=dict(size=16, color='rgb(0, 0, 0)'))
    traces.append(clean_points_scatter) # clean points scatter

    if nonmnfld_points is not None:
        # plot clean points and noisy points scatters
        nonmnfld_points_scatter = go.Scatter(x=nonmnfld_points[:, 0], y=nonmnfld_points[:, 1],
                                          mode='markers', marker=dict(size=16, color='rgb(255, 255, 255)'))
        traces.append(nonmnfld_points_scatter)  # clean points scatter

    # plot normal vectors:
    if n_gt is not None:
        f = ff.create_quiver(clean_points[:, 0], clean_points[:, 1], n_gt[:, 0], n_gt[:, 1], line=dict(color='rgb(0, 255, 0)'))
        traces.append(f.data[0])

    if grid_normals is not None:
        f = ff.create_quiver(x_grid, y_grid, grid_normals[:, 0], grid_normals[:, 1], line=dict(color='rgb(0, 0, 255)'),
                             scale=.01)
        traces.append(f.data[0])

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_sdf_indicator(points, x_grid, y_grid, sdf, title_text='', show_ax=False, output_path='./'):
    os.makedirs(output_path, exist_ok=True)
    points = np.concatenate([points, points[0, :][None, :]], axis=0)
    indicator = -np.ones_like(sdf)
    indicator[sdf > 0] = 1

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=True,
                                  mirror=True,  showline=True, linewidth=2, linecolor='black', showticklabels=False),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=True,
                                  mirror=True, showline=True,linewidth=2, linecolor='black', showticklabels=False),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red')),
                       paper_bgcolor='rgb(255,255,255)',
                       plot_bgcolor='rgb(255, 255, 255)'
                       )


    # plot clean points and noisy points scatters
    clean_points_scatter = go.Scatter(x=points[:, 0], y=points[:, 1],
                                mode='markers', marker=dict(size=52, color='rgb(0, 0, 0)'))
    fig = go.Figure(data=clean_points_scatter, layout=layout)
    fig.write_image(os.path.join(output_path, "points.png"))
    fig.data = []
    line_plot = go.Scatter(x=points[:, 0], y=points[:, 1],
                                mode='markers+lines', marker=dict(size=52, color='rgb(0, 0, 0)'),
                                      fill="toself", fillcolor='rgb(255,255,255)',
                                      line=dict(width=6, color='rgb(0, 0, 0)'))

    fig.add_traces(line_plot)
    fig.write_image(os.path.join(output_path, "lines.png"))
    fig.data = []

    sdf_trace = go.Contour(x=x_grid, y=y_grid, z=sdf, colorscale='Geyser',
               contours=dict(start=-1, end=1, size=0.025), showscale=False)
    sdf_zls_trace = go.Contour(x=x_grid, y=y_grid, z=sdf,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']])  # contour trace
    fig.add_traces([sdf_trace, sdf_zls_trace, clean_points_scatter])
    fig.write_image(os.path.join(output_path, "sdf.png"))
    fig.data = []

    indicator_trace = go.Contour(x=x_grid, y=y_grid, z=indicator, colorscale='Geyser',
               contours=dict(start=-1, end=1, size=0.025), showscale=False)
    indicator_zls_trace = go.Contour(x=x_grid, y=y_grid, z=indicator,
                           contours=dict(start=0, end=0, coloring='lines'),
                           line=dict(width=5),
                           showscale=False,
                           colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']])  # contour trace
    fig.add_traces([indicator_trace, indicator_zls_trace, clean_points_scatter])
    fig.write_image(os.path.join(output_path, "indicator.png"))

    # fig = go.Figure(data=traces, layout=layout)
    # fig.show()

def plot_distance_map(x_grid, y_grid, z_grid, title_text='', colorscale='Geyser', show_ax=True):
    # plot contour and scatter plot given input points

    traces = []
    # plot implicit function contour
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid, colorscale=colorscale,
                             contours=dict(start=-1, end=1, size=0.025), showscale=True))
    traces.append(go.Contour(x=x_grid, y=y_grid, z=z_grid,
                             contours=dict(start=0,  end=0, coloring='lines'),
                             line=dict(width=5),
                             showscale=False,
                             colorscale=[[0, 'rgb(100, 100, 100)'], [1, 'rgb(100, 100, 100)']]))  # contour trace

    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text=title_text, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def plot_points(points, title_txt='circle', example_idx=0, show_ax=True, color='rgb(0, 0, 0)', show=True, fig=None):
    points = points[example_idx]

    traces = []
    traces.append(go.Scatter(x=points[:, 0], y=points[:, 1],
                                mode='markers', marker=dict(size=16, color=color)))  # points scatter

    # plot jet
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-1, 1], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                  yaxis=dict(range=[-1, 1], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='Shape: ' + title_txt, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    if fig is None:
        fig = go.Figure(data=traces, layout=layout)
    else:
        fig.add_traces(traces)
    if show:
        fig.show()
    return fig

def plot_mesh(mesh_trace, mesh=None, output_ply_path=None, show_ax=False, title_txt='', show=True):
    # plot mesh and export ply file for mesh

    # export mesh
    if not(mesh is None or output_ply_path is None):
        mesh.export(output_ply_path, vertex_normal=True)

    # plot mesh
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                  yaxis=dict(range=[-2, 2], autorange=False),
                                  zaxis=dict(range=[-2, 2], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text='Shape: ' + title_txt, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    fig = go.Figure(data=mesh_trace, layout=layout)
    if show:
        fig.show()

def plot_sdf_surface(x, y, z, show=True, show_ax=True, title_txt=''):
    # plot mesh
    layout = go.Layout(width=1200, height=1200,
                       xaxis=dict(side="bottom", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       yaxis=dict(side="left", range=[-2, 2], showgrid=show_ax, zeroline=show_ax, visible=show_ax),
                       scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                  yaxis=dict(range=[-2, 2], autorange=False),
                                  zaxis=dict(range=[-2, 2], autorange=False),
                                  aspectratio=dict(x=1, y=1)),
                       showlegend=False,
                       title=dict(text= title_txt, y=0.95, x=0.5, xanchor='center', yanchor='middle',
                                  font=dict(family='Sherif', size=24, color='red'))
                       )
    surf_trace = go.Surface(z=z, x=x, y=y)
    fig = go.Figure(data=surf_trace, layout=layout)
    if show:
        fig.show()

    return fig


def plot_eval_metrics(metrics, data_range={'x':[0, 100], 'y':[0, 100]}, title_txt='', show=True, show_ax=True):
    ''' Plot evaluation metrics graphs'''
    out_img_dict = {}


    epochs = metrics['Epochs']
    for key in metrics.keys():
        plt.rcParams.update({'font.size': 32, 'lines.linewidth': 4,
                            #  'font.family': "Times New Roman"
                             })  # Set font size and line width
        if not key == 'Epochs':
            fig = plt.figure(1, (16, 16))
            canvas = fig.canvas
            plt.plot(epochs, metrics[key])
            plt.xlabel('Epoch')
            plt.ylabel(key)
            # set x range
            plt.xlim(data_range['x'])
            plt.ylim(data_range['y'])
            # save the image into a variable
            canvas.draw()
            image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            out_img_dict[key] = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
    plt.rcdefaults() # set params back to default
    return out_img_dict


def plot_rgb_experts(expert_idx, q, q_grad, q_raw=None, colormap='tab20', example_idx=0, n_cols=4, clim=(0.0, 0.5),
                 show_axis=False):


    n_experts = q.shape[0]
    n_rows = np.ceil(n_experts/n_cols).astype(int)
    width, height = expert_idx.shape
    extent = [0, width, 0, height]

    ################################### plot q heatmaps in a simgle image ###################################
    fig = plt.figure(1,(16, 16))
    canvas = fig.canvas
    axs = []
    for expert_id in range(n_experts):
        q_expert = q[expert_id]
        ax = fig.add_subplot(n_rows, n_cols, expert_id + 1)
        axs.append(ax)
        ax.set_title('Expert {} q'.format(expert_id))
        im = ax.imshow(q_expert, cmap='jet', interpolation='none',
                       extent=extent, clim=clim)

        # Set axis limits and labels
        plt.axis('off')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks(np.arange(extent[0], extent[1] + 0.5, 0.5))
        ax.set_yticks(np.arange(extent[2], extent[3] + 0.5, 0.5))
        # ax.set_aspect('equal')  # Ensure equal aspect ratio

    # Create a colorbar with discrete values
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02)
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    # plt.show()
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    q_image_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    ################################### plot q heatmaps in separate images ###################################
    q_image_array_list = []
    for expert_id in range(n_experts):
        q_expert = q[expert_id]
        fig = plt.figure(1, (16, 16))
        canvas = fig.canvas
        ax = fig.add_subplot(111)
        ax.imshow(q_expert, cmap='jet', interpolation='none',  extent=extent, clim=clim)
        # Set axis limits and labels
        plt.axis('off')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks(np.arange(extent[0], extent[1] + 0.5, 0.5))
        ax.set_yticks(np.arange(extent[2], extent[3] + 0.5, 0.5))
        # ax.set_aspect('equal')  # Ensure equal aspect ratio
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        q_image_array_list.append(image_array.reshape(canvas.get_width_height()[::-1] + (3,)))
        plt.close(fig)

    #################################### plot expert idx ####################################
    # Create a figure
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    # Create a custom colormap based on the unique values in expert_idx
    num_colors = n_experts
    custom_colormap = plt.get_cmap(colormap, num_colors)
    custom_norm = BoundaryNorm(np.arange(-0.5, num_colors, 1), custom_colormap.N)

    # Plot the image with the custom colormap
    im = ax.imshow(expert_idx, cmap=custom_colormap, norm=custom_norm, interpolation='none',
                   extent=extent)

    # Create a colorbar with discrete values
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(-0.5, num_colors, 1))
    # Set the tick labels to match the unique values in expert_idx
    values = range(n_experts)
    positions = [v for v in values]  # Centering the ticks
    cbar.set_ticks(positions)
    cbar.set_ticklabels(values)
    cbar.set_label('Expert ID')
    # plt.show()
    # Set axis limits and labels
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if show_axis:
        ax.set_xticks(np.arange(extent[0], extent[1]+0.5, 0.5))
        ax.set_yticks(np.arange(extent[2], extent[3]+0.5, 0.5))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_aspect('equal')  # Ensure equal aspect ratio
    fig.tight_layout()
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    experts_img = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    ############################# plot q gradient heatmaps #############################
    if q_grad is not None:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(q_grad, cmap='jet', interpolation='none',
                       extent=extent, clim=[0, 10])
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xticks(np.arange(extent[0], extent[1]+0.5, 0.5))
        ax.set_yticks(np.arange(extent[2], extent[3]+0.5, 0.5))
        # ax.set_aspect('equal')  # Ensure equal aspect ratio
        # Create a colorbar with discrete values
        cbar = plt.colorbar(im, ax=ax)
        # plt.show()
        canvas.draw()
        q_grad_image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        q_grad_image_array = q_grad_image_array.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        plt.close()
    else:
        q_grad_image_array = None

    ############################# plot q distribution per expert #############################
    qdist_image_array_list = []
    for expert_id in range(n_experts):
        q_expert = q[expert_id]
        fig = plt.figure(1, (16, 16))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # compute the histogram of q values
        frq, edges = np.histogram(q_expert.flatten(), bins=100)
        ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
        # plt.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5000)
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        qdist_image_array_list.append(image_array.reshape(canvas.get_width_height()[::-1] + (3,)))
        plt.close(fig)

    ################################### plot q distributions in a simgle image ###################################
    fig = plt.figure(1,(16, 16))
    canvas = fig.canvas
    axs = []
    for expert_id in range(n_experts):
        q_expert = q[expert_id]
        ax = fig.add_subplot(n_rows, n_cols, expert_id + 1)
        axs.append(ax)
        ax.set_title('Expert {} q distribution'.format(expert_id))
        frq, edges = np.histogram(q_expert.flatten(), bins=100)
        ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
        # Set axis limits and labels
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 5000)

    # Create a colorbar with discrete values
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    q_dist_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    q_raw_dist_array = None
    if q_raw is not None:
        ################################### plot q distributions in a simgle image ###################################
        fig = plt.figure(1, (16, 16))
        canvas = fig.canvas
        axs = []
        for expert_id in range(n_experts):
            q_expert = q_raw[expert_id]
            ax = fig.add_subplot(n_rows, n_cols, expert_id + 1)
            axs.append(ax)
            ax.set_title('Expert {} q distribution'.format(expert_id))
            frq, edges = np.histogram(q_expert.flatten(), bins=100)
            ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
            # Set axis limits and labels
            # ax.set_xlim(0, 1)
            # ax.set_ylim(0, 5000)

        # Create a colorbar with discrete values
        canvas.draw()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        q_raw_dist_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

    return experts_img, q_image_array, q_grad_image_array, q_image_array_list, qdist_image_array_list, q_dist_array, q_raw_dist_array


def plot_activations(activations, n_cols=4):
    n_layers = len(activations)
    n_rows = np.ceil(n_layers / n_cols).astype(int)

    ################################### plot activation distributions in a simgle image ###################################
    fig = plt.figure(1, (16, 16))
    canvas = fig.canvas
    axs = []
    max_freq = 0
    for layer_id in range(n_layers):
        layer_activations = activations[layer_id].detach().cpu().numpy()
        ax = fig.add_subplot(n_rows, n_cols, layer_id + 1)
        axs.append(ax)
        ax.set_title('layer {} ouptut distribution'.format(layer_id))
        frq, edges = np.histogram(layer_activations.flatten(), bins=100)
        ax.bar(edges[:-1], frq, width=np.diff(edges), edgecolor="black", align="edge")
        max_freq = max(max_freq, np.max(frq))
        # Set axis limits and labels
        # ax.set_xlim(0, 1)
    for ax in axs:
        ax.set_ylim(0, max_freq)

    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    activations_dist_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return activations_dist_array



################################### Audio visualizations ###################################


def plot_waveforms(coords, pred_waveform, gt_wavefrom):
    # plot waveforms.
    # Inspired by https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/utils.py#L532

    fig, axes = plt.subplots(3, 1)
    canvas = fig.canvas

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot].detach().cpu().numpy()

    gt_func_plot = gt_wavefrom.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_waveform.detach().cpu().numpy()[strt_plot:fin_plot]

    axes[1].plot(coords, pred_func_plot)
    axes[0].plot(coords, gt_func_plot)
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)

    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    waveform_img_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return waveform_img_array


def plot_audio_experts(coords, pred_waveform, gt_wavefrom, selected_expert_idx, q, q_grad, per_expert_recon_data):

    n_experts = q.shape[0]
    # plot waveforms for final prediction and ground truth
    fig, axes = plt.subplots(3, 1)
    canvas = fig.canvas

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot].detach().cpu().numpy()
    q = q[:, strt_plot:fin_plot].detach().cpu().numpy()
    # q_grad = q_grad[:, strt_plot:fin_plot]
    selected_expert_idx = selected_expert_idx[strt_plot:fin_plot].squeeze().detach().cpu().numpy()
    # per_expert_recon_data = [expert_recon[strt_plot:fin_plot].detach().cpu().numpy() for expert_recon in per_expert_recon_data]
    per_expert_recon_data = per_expert_recon_data[-1, :, strt_plot:fin_plot, ...].detach().cpu().numpy()

    gt_func_plot = gt_wavefrom.detach().cpu().numpy()[strt_plot:fin_plot]
    pred_func_plot = pred_waveform.detach().cpu().numpy()[strt_plot:fin_plot]

    cmap = cm.get_cmap('tab20')
    colors = cm.tab20(selected_expert_idx)

    segments = np.column_stack([coords[:-1], pred_func_plot[:-1], coords[1:], pred_func_plot[1:]])
    segments = segments.reshape(-1, 2, 2)
    lc = LineCollection(segments, colors=colors, linewidths=2, cmap=cmap)


    axes[0].plot(coords, gt_func_plot)
    # axes[1].plot(coords, pred_func_plot)
    # axes[0].set_ylim(-1, 1)
    axes[1].add_collection(lc)
    # axes[1].set_ylim(-1, 1)
    axes[1].autoscale()
    axes[2].plot(coords, gt_func_plot - pred_func_plot)

    # lc.set_clim(vmin=0, vmax=n_experts- 1)
    # Add horizontal colorbar at the bottom with specified ticks
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(n_experts + 1), n_experts), cmap=cmap),
                        ax=axes[2], orientation='horizontal', pad=0.1, ticks=np.arange(n_experts) + 0.5)
    cbar.set_ticklabels(np.arange(n_experts))  # Set tick labels to expert IDs
    cbar.set_label('Expert ID')


    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)
    axes[2].axes.get_xaxis().set_visible(False)
    # add colormap legend


    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    waveform_img_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)


    # plot q heatmaps on waveforms and expert data

    fig, axes = plt.subplots(n_experts+1, 1)
    canvas = fig.canvas
    axes[0].plot(coords, gt_func_plot)
    axes[0].get_xaxis().set_visible(False)
    for i in range(n_experts):
        # axes[i+1].plot(coords, expert_recon)
        colors = cm.jet(q[i, ...])

        segments = np.column_stack([coords[:-1], per_expert_recon_data[i][:-1], coords[1:], per_expert_recon_data[i][1:]])
        segments = segments.reshape(-1, 2, 2)
        lc = LineCollection(segments, colors=colors, linewidths=2, clim=(0, 1))

        axes[i+1].add_collection(lc)
        # axes[i+1].set_ylim(-1, 1)
        axes[i+1].autoscale()
        axes[i+1].get_xaxis().set_visible(False)


    cmap = cm.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=1)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes[i+1], orientation='horizontal', pad=0.05)
    # cbar.set_ticklabels(np.arange(n_experts)) # Set tick labels to expert IDs
    cbar.set_label('Expert q value')

    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    waveform_img_per_expert_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return waveform_img_array, waveform_img_per_expert_array



def plot_audio_waveform_w_segments(coords, gt_wavefrom, segments):

    n_segments = len(np.unique(segments))
    # plot waveforms for final prediction and ground truth
    fig, axes = plt.subplots(2, 1)
    canvas = fig.canvas

    strt_plot, fin_plot = int(0.05*len(coords)), int(0.95*len(coords))
    coords = coords[strt_plot:fin_plot].detach().cpu().numpy()
    segments = segments.flatten()[strt_plot:fin_plot]

    gt_func_plot = gt_wavefrom[strt_plot:fin_plot]

    cmap = cm.get_cmap('tab20')
    colors = cm.tab20(segments)

    plot_segments = np.column_stack([coords[:-1], gt_func_plot[:-1], coords[1:], gt_func_plot[1:]])
    plot_segments = plot_segments.reshape(-1, 2, 2)
    lc = LineCollection(plot_segments, colors=colors, linewidths=2, cmap=cmap)


    axes[0].plot(coords, gt_func_plot)
    axes[1].add_collection(lc)
    axes[1].autoscale()

    # Add horizontal colorbar at the bottom with specified ticks
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(n_segments + 1), n_segments), cmap=cmap),
                        ax=axes[1], orientation='horizontal', pad=0.1, ticks=np.arange(n_segments) + 0.5)
    cbar.set_ticklabels(np.arange(n_segments))  # Set tick labels to expert IDs
    cbar.set_label('Segment ID')

    axes[0].get_xaxis().set_visible(False)
    axes[1].axes.get_xaxis().set_visible(False)

    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    waveform_img_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return waveform_img_array

def plot_audio_spectrograms_w_segments(coords, gt_wavefrom, rate, segments):
    powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(gt_wavefrom, Fs=rate)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


