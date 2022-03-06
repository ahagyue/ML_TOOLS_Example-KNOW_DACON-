import matplotlib.pyplot as plt
import numpy as np


'''
error message!!
'''
def call_argument_exception():
    print("Err : Argument Exception")
def call_plot_size_exception():
    print("Err : Plot Size Exception")
def call_type_error():
    print("Err: Type Error")
def call_out_of_range_error():
    print("Err : Out of Range")



'''
functions

- visualize_graph(x_value , y_value, graph_name, y_lim, ax, title, x_label, y_label)
    -> x_value : np.array, x_value.ndim >= 1
    -> y_value : np.array, y_value.ndim >= 1
        if x_value.ndim ==1 : 2 >= y_value.ndim >=1, y_value.shape[-1] == x_value.shape[0]
        else : x_value.shape[0] == y_value.shape[0]
    -> graph_name.ndim == y_value.ndim
    -> y_lim : tuple

-visualize_image(image, ax, title)
    -> image : np.array(2D)
    -> ax : sketch book
    -> title : title of image

class
- Plots : for subplot
    -functions
    -> append_graph : append graph
    -> append_img : append image
    -> out_of_range
    -> repair_to_graph : repair to graph
    -> repair_to_img 
'''


def visualize_graph(
        x_value : np.ndarray, y_value : np.ndarray, graph_name : list = None,
        y_lim : tuple = None, ax = None, title : str = None, x_label : str = None, y_label : str = None
):
    if ax is None: fig, ax = plt.subplot(figsize=(10,10))
    if y_value.ndim == 1 :
        if x_value.ndim > 1 or y_value.shape[0] != x_value.shape[0]:
            call_argument_exception(); return
        ax.plot(x_value, y_value)
    
    elif y_value.ndim == 2:
        if x_value.ndim > 2 or x_value.shape[-1] != y_value.shape[-1]:
            call_argument_exception(); return
        if x_value.ndim == 1:
            for i in range(y_value.shape[0]):
                ax.plot(x_value, y_value[i])
        else:
            for i in range(y_value.shape[0]):
                ax.plot(x_value[i], y_value[i])
    
    if graph_name != None:
        ax.legend(graph_name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_ylim(y_lim)


def visualize_image(image, ax = None, title : str = None):
    if(ax == None): fig, ax = plt.subplot(figsize=(10,10))
    try: im = ax.imshow(image)
    except:
        call_type_error()
        return
    plt.colorbar(im, ax=ax, format='%.5f')
    ax.set_title(title)







class Plots:
    def __init__(self, fig=None, subplot_num = 1, position : tuple = (1,1), suptitle : str = None):
        if fig == None:
            self.fig = plt.figure()
        else:
            self.fig = fig
        self.ax = []
        self.subplot_num = subplot_num

        if(position[0]*position[1] != subplot_num):
            call_plot_size_exception()
            return
        
        for i in range(1, subplot_num+1):
            try: self.ax.append(self.fig.add_subplot(position[0], position[1], i))
            except:
                call_plot_size_exception()
                return
        
        self.current_fig = 0
        self.st=fig.suptitle(suptitle, fontsize='xx-large')

        print("Succeed making sketch book")
    
    def out_of_range(self):
        return self.subplot_num <= self.current_fig

    def append_graph(
            self,
            x_value : np.ndarray, y_value : np.ndarray, graph_name : list = None,
            y_lim : tuple = None, title : str = None, x_label : str = None, y_label : str = None
    ):
        if self.out_of_range():
            call_out_of_range_error()
            return
        visualize_graph(x_value, y_value, graph_name, y_lim, self.ax[self.current_fig], title, x_label, y_label)
        self.current_fig += 1

    def append_img(self, image, title : str = None):
        if self.out_of_range():
            call_out_of_range_error()
            return
        visualize_image(image, self.ax[self.current_fig], title)
        self.current_fig += 1
    
    def repair_to_graph(
            self, plot_num : int,
            x_value : np.ndarray, y_value : np.ndarray, graph_name : list = None,
            y_lim : tuple = None, title : str = None, x_label : str = None, y_label : str = None
    ):
        if self.subplot_num <= plot_num:
            call_out_of_range_error()
            return
        self.ax[plot_num].clear()
        visualize_graph(x_value, y_value, graph_name, y_lim, self.ax[plot_num], title, x_label, y_label)
        
    def repair_to_img(self, plot_num : int, image, title : str = None):
        if self.subplot_num <= plot_num:
            call_out_of_range_error()
            return
        self.ax[plot_num].clear()
        
        visualize_image(image, self.ax[plot_num], title)
    
    def return_suptitle(self):
        return self.st



