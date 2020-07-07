#!/usr/bin/env python
#
# Copyright 2007 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import cgi
import datetime
import webapp2
import logging

from google.appengine.ext import ndb
from google.appengine.api import users
from google.appengine.api import mail
from google.appengine.api import urlfetch

import os
import jinja2
import urllib, hashlib
import json

TEMPLATEFOLDER = "_build/html/"

JINJA_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(
        os.path.join(os.path.dirname(__file__).split("/")[:-1])
    ),
    extensions=["jinja2.ext.autoescape"],
    autoescape=False,
)


def setTemplate(self, template_values, templateFile, _templateFolder=TEMPLATEFOLDER):
    # add Defaults
    template_values["_templateFolder"] = _templateFolder
    template_values["_year"] = str(datetime.datetime.now().year)
    path = os.path.normpath(_templateFolder + templateFile)
    template = JINJA_ENVIRONMENT.get_template(path)
    resp = self.response.write(template.render(template_values))

    # if resp is None:
    #     self.redirect('/error.html', permanent=True)


class Images(webapp2.RequestHandler):
    def get(self):
        self.redirect("/" + self.request.path)


class Redirect(webapp2.RequestHandler):
    def get(self):
        path = str(self.request.path).split(os.path.sep)[3:]
        self.redirect(("/{0!s}".format(os.path.sep.join(path))), permanent=True)


class MainPage(webapp2.RequestHandler):
    def get(self):
        setTemplate(self, {"indexPage": True}, "index.html")


# class Error(webapp2.RequestHandler):
#     def get(self):
#         setTemplate(self, {}, 'error.html', _templateFolder='_templates/')
#         # self.redirect('/error.html', permanent=True)

from webapp2 import Route, RedirectHandler

# pointers = [
#             Route('/en/latest/.*', RedirectHandler, defaults={'_uri': '/.*'}),
#             # Route('/en/latest', RedirectHandler, defaults={'_uri': '/en/latest/'}),
#             Route('/en/latest/', RedirectHandler, defaults={'_uri': '/'}),
#
#             ('/.*', MainPage),
#             ('/', MainPage),
#             ('', MainPage),
#             ('/_images/.*', Images),
#             ]

app = webapp2.WSGIApplication(
    [
        ("/_images/.*", Images),
        ("/en/latest/.*", Redirect),
        ("/", MainPage),
        # ('/.*', Error),
    ],
    debug=True,
)


# app.error_handlers[404] = Error
