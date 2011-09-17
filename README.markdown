This module provides a way to annotate multiple points. Annotating texts
are drawn along a (vertical or horizontal) line. And the offset
between texts are automatically adjusted so that they do not overwrap.

![annotation_set sample](https://raw.github.com/leejjoon/mpl_annotation_set/master/doc/_static/example.png)

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

