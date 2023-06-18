import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import math
import torch
import torch.nn as nn
import torchvision
def plot_lnK(Xm,Ym,lnK,Kbound=None,measure_id=None,reshape_order='C', title="lnK field", cmap="plasma"):
    nx = Xm.shape[-2]
    ny = Xm.shape[-1]
    fig2,ax2 = plt.subplots(figsize=(4,3))
    if Kbound is None:
        minlK, maxlK = np.min(lnK), np.max(lnK)
        Kbound = (minlK, maxlK)
    im2 = ax2.pcolormesh(Xm,Ym,lnK.reshape((nx,ny),order=reshape_order), cmap=cmap)
    im2.set_clim(*Kbound)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig2.colorbar(im2, ax=ax2, orientation='vertical')
    if measure_id is not None:
        ax2.scatter(Xm.flatten()[measure_id], Ym.flatten()[measure_id], marker='v', zorder=1, alpha= 1, c='m', s=10)
    ax2.set_title(title)



def plot_fields(preds, refs, add_contour=False, cmap="plasma"):
    axis_label_font_size = 15
    axis_tick_font_size = 15
    legend_fontszie = 25
    colorbar_font_size = 15
    title_size = 15
    contour_size = 10
    shift=0
    
    nx = preds.shape[-1]
    ny = preds.shape[-1]
    x = np.linspace(-0.5,0.5,nx)
    y = np.linspace(-0.5,0.5,ny)
    Xm, Ym = np.meshgrid(x,y)
    
    hmin, hmax = np.min(preds,axis=(1,2)), np.max(preds,axis=(1,2))

    hmin *= 1.0
    hmax *= 1.0

    gridspec_kw=dict(wspace=0.6,hspace=0.5)
    fig, axs = plt.subplots(2, 5,figsize=(22,7), gridspec_kw=gridspec_kw)
    axs = axs.flatten()

    for iid in range(5):
        ax = axs[iid]

        Zm = preds[iid,...]
        im = ax.pcolormesh(Xm, Ym, Zm, vmin=hmin[iid], vmax=hmax[iid], cmap=cmap)

        ax.set_title('Pred%d' % (iid), fontsize=title_size)
        make_color_bar(fig, im, ax, hmin[iid], hmax[iid])

        if add_contour:
            lvls = np.linspace(hmin[iid], hmax[iid],7)
            CP = ax.contour(Xm, Ym, Zm, levels=lvls,cmap="coolwarm")
            ax.clabel(CP,fontsize=contour_size,inline=True,inline_spacing=1,fmt='%.2f')
        
        ax = axs[iid+5]
        Zm = refs[iid,...]
        im = ax.pcolormesh(Xm, Ym, Zm, vmin=hmin[iid], vmax=hmax[iid], cmap=cmap)
        ax.set_title('True%d' % (iid), fontsize=title_size)
        make_color_bar(fig, im, ax, hmin[iid], hmax[iid])

        if add_contour:
            lvls = np.linspace(hmin[iid], hmax[iid],7)
            CP = ax.contour(Xm, Ym, Zm, levels=lvls,cmap="coolwarm")
            ax.clabel(CP,fontsize=contour_size,inline=True,inline_spacing=1,fmt='%.2f')

    for aid in range(10):
        ax = axs[aid]
        ######### x-axis name, ticks and labels #########
        ticks = [-0.5, 0.0, 0.5]
        labels = [0, 0.5, 1]
        ax.set_xlabel('x',fontsize=axis_label_font_size)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,fontsize=axis_tick_font_size,ha='center')

        ######### y-axis name, ticks and labels #########
        ax.set_ylabel('y',fontsize=axis_label_font_size)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels,fontsize=axis_tick_font_size,rotation=90, ha='right', va='center')
    return fig
       
def smooth_img(imgs, in_channels=1, cmap='plasma', show_img=False):        
    # compute the weights of the filter with the given size (and additional params)
    h_kernel = torch.tensor([[0, 0, 0],[2/9, 5/9, 2/9],[0, 0, 0]],dtype=torch.float32) 
    h_kernel = h_kernel.view(1,1,3,3).repeat(1,in_channels,1,1)
    h_layer = nn.Conv2d(1,1,3, bias=False, padding='same')
    h_layer.weight = nn.Parameter(h_kernel,requires_grad=False)
    
    # compute the weights of the filter with the given size (and additional params)
    v_kernel = torch.tensor([[0, 2/9, 0],[0, 5/9, 0],[0, 2/9, 0]],dtype=torch.float32)
    v_kernel = v_kernel.view(1,1,3,3).repeat(1,in_channels,1,1)
    v_layer = nn.Conv2d(1,1,3, bias=False, padding='same')
    v_layer.weight = nn.Parameter(v_kernel,requires_grad=False)
    if show_img:
        fig,(ax1, ax2) = plt.subplots(1,2,figsize=(10,3))
        
        im = ax1.imshow(imgs[0,0,:,:].detach().cpu().numpy(), cmap=cmap)
        cbar = fig.colorbar(im, ax=ax1, shrink=0.95)
        ax1.set_title("Original")
    
    for i in range(3):      
      imgs = h_layer(imgs)
      imgs= v_layer(imgs)
      
    if show_img:
        im = ax2.imshow(imgs[0,0,:,:].detach().cpu().numpy(), cmap=cmap)
        ax2.set_title("Smoothed")
        cbar = fig.colorbar(im, ax=ax2, shrink=0.95)
    
    return imgs

def show_imgs(imgs, title=None, row_size=4):
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
    is_int = imgs.dtype==torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype==torch.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
        
def make_color_bar(fig, im, ax, zmin, zmax, colorbar_font_size=15):
    # Divide existing axes and create new axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="8%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax,ticks=[round(x,2) for x in\
                                          np.arange(zmin, zmax,(zmax-zmin)/3)])
    cbar.ax.tick_params(labelsize=colorbar_font_size) 

def plot_inverse_operator(hs, preds, refs, cmaplnK='jet'):
      
    #set font size
    axis_label_font_size = 15
    axis_tick_font_size = 15
    legend_fontszie = 15
    colorbar_font_size = 15
    title_size = 20
    contour_size = 10
    
    
    nx = preds.shape[-1]
    ny = preds.shape[-1]
    Xm, Ym = np.meshgrid(np.linspace(-0.5,0.5,nx),np.linspace(-0.5,0.5,ny))
    
    
    # image grids
    gridspec_kw=dict(wspace=0.55,hspace=0.5)
    fig, axs = plt.subplots(3, 5, figsize=(25,13), gridspec_kw=gridspec_kw)
    axs = axs.flatten()

    predmin, predmax = np.min(preds,axis=(1,2)), np.max(preds,axis=(1,2))
    # predmin *= 0.950
    # predmax *= 1.00

    for iid in range(5):
        # ax[0 - 4] heads map of one pumping test in 5 inverse cases
        ax = axs[iid]
        Zm = hs[iid].reshape((nx,ny))
        im = ax.pcolormesh(Xm,Ym,Zm,cmap='viridis')
        ax.set_title('(A%d) Input Heads %d' % (iid+1,iid+1), fontsize=title_size)

        make_color_bar(fig, im, ax, np.min(Zm)*0.95, np.max(Zm)*1.0)

        lvls = np.linspace(np.min(Zm)*0.95, np.max(Zm)*1.0, 7)
        CP = ax.contour(Xm, Ym, Zm, levels=lvls, cmap="coolwarm")
        ax.clabel(CP,fontsize=contour_size,inline=True,inline_spacing=1,fmt='%.2f')

        # ax[5 - 9] predcited channel fields
        ax = axs[iid+5]
        Zm = preds[iid].reshape((nx,ny))
        im = ax.pcolormesh(Xm,Ym,Zm,cmap=cmaplnK)
        ax.set_title('(B%d) Pred %d' % (iid+1,iid+1), fontsize=title_size)
        make_color_bar(fig, im, ax, predmin[iid], predmax[iid])

        # ax[10 - 14] true channel fields
        ax = axs[iid+10]
        Zm = refs[iid].reshape((nx,ny))
        im = ax.pcolormesh(Xm,Ym,Zm,cmap=cmaplnK)
        ax.set_title('(C%d) True %d' % (iid+1,iid+1), fontsize=title_size)
        make_color_bar(fig, im, ax, predmin[iid], predmax[iid])

    for aid in range(15):
        ax = axs[aid]
        ######### x-axis name, ticks and labels #########
        ticks = [-0.5, 0.0, 0.5]
        labels = [0, 0.5, 1]
        ax.set_xlabel('x',fontsize=axis_label_font_size)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,fontsize=axis_tick_font_size,ha='center')

        ######### y-axis name, ticks and labels #########
        ax.set_ylabel('y',fontsize=axis_label_font_size)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels,fontsize=axis_tick_font_size,rotation=90, ha='right', va='center')

    return fig

def plot_forward_operator(lnKs, preds, refs, cmaplnK='jet'):
      
    #set font size
    axis_label_font_size = 15
    axis_tick_font_size = 15
    legend_fontszie = 15
    colorbar_font_size = 15
    title_size = 20
    contour_size = 10
    
    nx = preds.shape[-1]
    ny = preds.shape[-1]
    Xm, Ym = np.meshgrid(np.linspace(-0.5,0.5,nx),np.linspace(-0.5,0.5,ny))
    
    # image grids
    gridspec_kw=dict(wspace=0.55,hspace=0.5)
    fig, axs = plt.subplots(3, 5, figsize=(25,13), gridspec_kw=gridspec_kw)
    axs = axs.flatten()

    predmin, predmax = np.min(preds,axis=(1,2)), np.max(preds,axis=(1,2))
    predmin *= 0.950
    predmax *= 1.00

    for iid in range(5):
        # ax[0 - 4] heads map of one pumping test in 5 inverse cases
        ax = axs[iid]
        Zm = lnKs[iid].reshape((nx,ny))
        im = ax.pcolormesh(Xm,Ym,Zm,cmap=cmaplnK)
        ax.set_title('(A%d) Input Heads %d' % (iid+1,iid+1), fontsize=title_size)

        make_color_bar(fig, im, ax, np.min(Zm)*0.95, np.max(Zm)*1.0)

        # ax[5 - 9] predcited channel fields
        ax = axs[iid+5]
        Zm = preds[iid].reshape((nx,ny))
        im = ax.pcolormesh(Xm,Ym,Zm,cmap='viridis')
        ax.set_title('(B%d) Pred %d' % (iid+1,iid+1), fontsize=title_size)
        make_color_bar(fig, im, ax, predmin[iid], predmax[iid])

        lvls = np.linspace(predmin[iid], predmax[iid], 7)
        CP = ax.contour(Xm, Ym, Zm, levels=lvls, cmap="coolwarm")
        ax.clabel(CP,fontsize=contour_size,inline=True,inline_spacing=1,fmt='%.2f')

        # ax[10 - 14] true channel fields
        ax = axs[iid+10]
        Zm = refs[iid].reshape((nx,ny))
        im = ax.pcolormesh(Xm,Ym,Zm,cmap='viridis')
        ax.set_title('(C%d) True %d' % (iid+1,iid+1), fontsize=title_size)
        make_color_bar(fig, im, ax, predmin[iid], predmax[iid])

        lvls = np.linspace(predmin[iid], predmax[iid], 7)
        CP = ax.contour(Xm, Ym, Zm, levels=lvls, cmap="coolwarm")
        ax.clabel(CP,fontsize=contour_size,inline=True,inline_spacing=1,fmt='%.2f')

    for aid in range(15):
        ax = axs[aid]
        ######### x-axis name, ticks and labels #########
        ticks = [-0.5, 0.0, 0.5]
        labels = [0, 0.5, 1]
        ax.set_xlabel('x',fontsize=axis_label_font_size)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,fontsize=axis_tick_font_size,ha='center')

        ######### y-axis name, ticks and labels #########
        ax.set_ylabel('y',fontsize=axis_label_font_size)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels,fontsize=axis_tick_font_size,rotation=90, ha='right', va='center')

    return fig
