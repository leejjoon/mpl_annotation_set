from distutils.core import setup

from distutils.command.install import install as _install

import os

setup(
  name              = "annotation_set",
  version           = "0.1",
  description       = "",
  long_description  = """""",
  url               = "",
  download_url      = "",
  author            = "Jae-Joon Lee",
  author_email      = "lee.j.joon@gmail.com",
  platforms         = ["any"],
  license           = "MIT",
  packages          = ['mpl_toolkits'],
  package_dir       = {'mpl_toolkits':'mpl_toolkits',
                       #"mpl_toolkits.annotation_set":"mpl_toolkits/annotation_set",
                       }
  )
