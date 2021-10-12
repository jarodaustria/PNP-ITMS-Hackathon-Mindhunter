"""
WSGI config for pnp_itms_hackathon_mindhunters project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pnp_itms_hackathon_mindhunters.settings')

application = get_wsgi_application()
