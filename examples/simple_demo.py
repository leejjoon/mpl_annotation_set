
from mpl_annotation_set.annotation_set import axes_annotation_set, DefaultConnectionStyleFunc, OffsetBy


if 1:

    import matplotlib.pyplot as plt
    import numpy as np
    
    x = np.random.rand(10)
    y = np.random.rand(10)
    xy_list = np.vstack([x, y]).T # zip(x, y)
    
    s_list = ["$000%d$" % i for i in range(len(x))]
    s_list[3] = "a"
    

    ax = plt.subplot(121)
    ax.plot(x, y, "o")
    
    att = axes_annotation_set(ax, "top",
                              s_list, xy_list, 
                              xycoords='data',
                              refpos=1.3,
                              refcoord="data"
                              )

    ax.set(xlim=(-1., 1.5), ylim=(-0.5, 1.5))


    ax = plt.subplot(122)
    ax.plot(x, y, "o")
    
    att = axes_annotation_set(ax, "right",
                              s_list, xy_list, 
                              xycoords='data',
                              refpos=-1,
                              # refcoord="axes fraction",
                              connection_style_func=DefaultConnectionStyleFunc(armA=8, rad=2)
                              )

    refcoord=OffsetBy((0.95, 0.95), ax.transAxes, att.get_max_width_height)
    att.set_refcoord(refcoord)
    
    ax.set(xlim=(-0.5, 2.), ylim=(-0.5, 1.5))

    plt.show()
    
