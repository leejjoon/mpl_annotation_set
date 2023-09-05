"""
This module provides a way annotate multiple points. Annotating texts
are drawn along a single (vertical or horizontal) line. And the offset
between texts are automatically adjusted so that they do not overwrap.

Running this command is similar to *annotate*. For example,:

    s_list = ["label1", "label2", "label3"]
    xy_list = [(0, 0), (1, 2), (2, 1)]
    direction = "top"

    att = axes_annotation_set(ax, direction,
                              s_list, xy_list,
                              xycoords='data',
                              refpos=1.3,
                              refcoord="data"
                              )

Here, s_list is a list of labels and the xy_list is a list of xy
positions to be annotated. *xycoords* has same meaning as in the
*annotate* and it aplies to position in *xy_list*. The *direction*
argument specifies the direction of annotated texts ("top",
"bottom", "left", or "right"). *refpos* and *refcoord* is
position and coordinate for annotating
texts. Unlike *annotate*, you must specify a single value for the
position. For direction of"top" and "bottom", the *refpos* means y-position of
annoatating texts and x-positions are automatically determined. For
direction of "left" and "right", the *refpos* means x-position of
texts and y-positions are automatically adjusted.
"""

import matplotlib.transforms as mtransforms
from matplotlib.text import Annotation

from scipy.optimize import fmin_cobyla
import numpy as np




def get_objective_constraints(x1, dx):

    # minimize sum of the distance square
    def objective(x):
        return ((x-x1)**2).sum()

    # constraints
    constraints = [lambda x, n=i: x[n+1] - x[n] - .5*(dx[n] + dx[n+1]) for i in range(len(x1)-1)]

    return objective, constraints


def optimize_spacing(x0, dx):
    x0 = np.asarray(x0)
    sorted_index = np.argsort(x0)

    x1 = x0[sorted_index]
    dx = np.asarray(dx)[sorted_index]

    objective, constraints = get_objective_constraints(x1, dx)

    x1_optimized = fmin_cobyla(objective, x1, constraints, rhoend=1e-7)

    x0_optimized = np.empty_like(x1_optimized)
    x0_optimized[sorted_index] = x1_optimized

    return x0_optimized


def test_optimize():
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    xx = np.random.rand(10)
    xx3 = optimize_spacing(xx, [0.05]*len(xx))

    plt.plot(xx, np.zeros_like(xx), "o")
    plt.ylim(-0.1, 1.5)

    for x1, x3 in zip(xx, xx3):
        plt.annotate(str(x1), (x1, 0), xytext=(x3, 1), xycoords='data',
                     textcoords='data',
                     rotation=90, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="-",
                                     relpos=(0.5, 0.),
                                     #connectionstyle="arc3"))
                                     connectionstyle="arc,angleA=270,angleB=90,armA=50,armB=50,rad=10"))



class DefaultConnectionStyleFunc(object):
    def __init__(self, armA=20, rad=3, pad=None):
        self.rad = rad
        self.armA = armA
        self.pad = pad

    def __call__(self, fontsize_in_pixel, angleA, angleB, dy):
        rad, armA = self.rad, self.armA
        if self.pad is None:
            pad = rad*2
        else:
            pad = self.pad
        connectionstyle = "arc"
        connectionstyle_attr = dict(angleA=angleA,angleB=angleB,
                                    armA=armA,armB=dy-armA-pad,rad=rad)

        return connectionstyle, connectionstyle_attr


class AnnotationSet(Annotation):

    _default_values = dict(left=dict(INDX=0, ha="right", va="center",
                                     relpos=(1., 0.5), angleAangleB=(0, 180), rot=0),
                           right=dict(INDX=0, ha="left", va="center",
                                      relpos=(0., 0.5), angleAangleB=(180, 0), rot=0),
                           bottom=dict(INDX=1, ha="center", va="top",
                                       relpos=(0.5, 1.), angleAangleB=(90, 270), rot=270),
                           top=dict(INDX=1, ha="center", va="bottom",
                                    relpos=(0.5, 0.), angleAangleB=(270, 90), rot=90),
                           )

    def __init__(self, direction,
                 s_list, xy_list,
                 refpos,
                 xycoords='data',
                 refcoord='data',
                 arrowprops=None,
                 annotation_clip=None,
                 pad=0.15,
                 patches=None,
                 connection_style_func=DefaultConnectionStyleFunc(),
                 **kwargs):


        if direction not in self._default_values.keys():
            raise ValueError("direction must be one of %s" + ",".join(self._default_values.keys()))

        self._direction = direction
        default_values=self._default_values[direction]

        INDX = default_values["INDX"]

        if not ("va" in kwargs or "verticalalignment" in kwargs):
            kwargs["va"] = default_values["va"]

        if not ("ha" in kwargs or "horizontalalignment" in kwargs):
            kwargs["ha"] = default_values["ha"]

        if not "rotation" in kwargs:
            kwargs["rotation"] = default_values["rot"]

        if refcoord not in ["data", "axes fraction", "figure fraction"]:
            raise ValueError("")

        self._s_list = s_list
        self._xy_list = xy_list
        self._pad = pad

        xytext = [0, 0]
        xytext[INDX] = refpos

        textcoords = ["figure pixels", "figure pixels"]
        textcoords[INDX] = refcoord

        self._INDX = INDX


        self.arrowprops = arrowprops

        self.arrow = None

        if arrowprops is None:
            arrowprops = dict(arrowstyle="-", relpos=default_values["relpos"])
        elif arrowprops.has_key("arrowstyle") and "relpos" not in arrowprops:
            arrowprops["relpos"] = default_values["relpos"]

        self._connection_style_func = connection_style_func

        Annotation.__init__(self, "", (0,0),
                            xytext=xytext,
                            xycoords=xycoords,
                            textcoords=tuple(textcoords),
                            arrowprops=arrowprops,
                            annotation_clip=annotation_clip,
                            **kwargs)


    def set_refcoord(self, refcoord):
        textcoords = ["figure pixels", "figure pixels"]
        textcoords[self._INDX] = refcoord

        self.anncoords = tuple(textcoords)


    def get_widths_heights(self, renderer):


        w_list, h_list = [], []
        old_s = self.get_text()
        for s in self._s_list:
            self.set_text(s)
            # get_window_extent returns a bbox including arrow
            # bbox = super(AnnotationSet, self).get_window_extent(renderer)
            bbox, info, descent = self._get_layout(renderer)
            x, y, w, h = bbox.bounds
            w_list.append(w)
            h_list.append(h)

        self.set_text(old_s)

        return w_list, h_list


    def get_max_width_height(self, renderer):

        w_list, h_list = self.get_widths_heights(renderer)
        return max(w_list), max(h_list)


    #@allow_rasterization
    def draw(self, renderer):
        """
        Draw the :class:`Annotation` object to the given *renderer*.
        """

        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible(): return

        default_values=self._default_values[self._direction]
        angleA, angleB = default_values["angleAangleB"]

        vis_list = []
        x_pix_list = []
        dy_pix_list = []

        for x,y in self._xy_list:
            self.xy = (x, y)
            xy_pixel = self._get_position_xy(renderer)
            # self._update_position_xytext(renderer, xy_pixel)
            self.update_positions(renderer)
            #xytext = self._x, self._y
            xy = self._get_xy_display()
            # dy_pix_list.append(abs([self._x, self._y][self._INDX] - xy_pixel[self._INDX]))
            dy_pix_list.append(abs(xy[self._INDX] - xy_pixel[self._INDX]))
            # dy_pix_list.append(100)
            ###
            x_pix_list.append(xy_pixel[1-self._INDX])

            if self._check_xy(renderer):
                vis_list.append(True)
            else:
                vis_list.append(False)

        vis_list = np.array(vis_list)
        #####
        # update x_pix_list
        fontsize_in_pixel = renderer.points_to_pixels(self.get_size())

        pad = fontsize_in_pixel*self._pad
        wh_list = self.get_widths_heights(renderer)
        size_list = [s+pad for s in wh_list[1-self._INDX]]

        if len(vis_list):
            x_pix_list_masked = np.array(x_pix_list)[vis_list]
            x_pix_list0_masked = optimize_spacing(x_pix_list_masked, size_list)

            for s, (x,y), x2, dy in zip(np.array(self._s_list)[vis_list],
                                        np.array(self._xy_list)[vis_list],
                                        x_pix_list0_masked,
                                        np.array(dy_pix_list)[vis_list]):

                self.xy = (x, y)
                # self.xytext[1-self._INDX] = x2
                # self.xytext[1-self._INDX] = x2
                xytext = list(self.get_position()) # tuple to list
                xytext[1-self._INDX] = x2
                self.xyann = xytext
                self.set_text(s)

                if self._connection_style_func:
                    cs, cs_attr = self._connection_style_func(fontsize_in_pixel,
                                                              angleA, angleB,
                                                              dy)

                    self.arrow_patch.set_connectionstyle(cs, **cs_attr)
                                                         # angleA=angleA,angleB=angleB,
                                                         # armA=20,armB=dy-20-rad*2,rad=rad)

                super().draw(renderer)

        #for s, (x, y), x2, dy in self._loop_over(renderer):


def axes_annotation_set(ax, direction,
                        s_list, xy_list,
                        refpos,
                        xycoords='data',
                        refcoord='data',
                        arrowprops=None,
                        annotation_clip=None,
                        pad=0.5,
                        **kwargs):

    att = AnnotationSet(direction,
                        s_list, xy_list,
                        refpos,
                        xycoords=xycoords,
                        refcoord=refcoord,
                        arrowprops=arrowprops,
                        annotation_clip=annotation_clip,
                        pad=pad,
                        **kwargs)

    att.set_transform(mtransforms.IdentityTransform())

    att.set_clip_on(False)
    ax._set_artist_props(att)
    # ax.artists.append(att)
    ax.add_artist(att)

    return att


from matplotlib.artist import Artist
from matplotlib.transforms import Affine2D, BboxBase

class OffsetBy(object):
    def __init__(self, ref_pos, ref_transform, unit):
        self._ref_pos = ref_pos
        self._ref_transform = ref_transform
        self._unit = unit

        #self.set_unit(unit)

    def __call__(self, renderer):
        if isinstance(self._unit, Artist):
            bbox = self._unit.get_window_extent(renderer)
            l, b, w, h = bbox.bounds
        elif isinstance(self._unit, BboxBase):
            l, b, w, h = self._unit.bounds
        elif callable(self._unit):
            w, h = self._unit(renderer)
        else:
            raise RuntimeError("unknown type")

        x, y = self._ref_transform.transform_point(self._ref_pos)
        tr = Affine2D().scale(w, h).translate(x, y)

        return tr


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.random.rand(10)
    y = np.random.rand(10)
    xy_list = zip(x, y)

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
                              connection_style_func=DefaultConnectionStyleFunc(armA=8, rad=2)
                              )

    refcoord=OffsetBy((0.95, 0.95), ax.transAxes, att.get_max_width_height)
    att.set_refcoord(refcoord)

    ax.set(xlim=(-0.5, 2.), ylim=(-0.5, 1.5))

    plt.show()

