from django.contrib import admin
from .models import Station, CCTV, Crime

from django.urls import reverse
from django.utils.http import urlencode
from django.utils.html import format_html

# Register your models here.
@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ("name","city","view_cctvs_link")
    search_fields = ("name__startswith",)

    def view_cctvs_link(self,obj):
        count = obj.cctv_set.count()
        url = (
            reverse("admin:webcam_cctv_changelist")
            + "?"
            + urlencode({"stations__id": f"{obj.id}"})
        )
        return format_html('<a href="{}">{} CCTVs</a>', url, count)

    view_cctvs_link.short_description = "CCTVs"

@admin.register(CCTV)
class CCTVAdmin(admin.ModelAdmin):
    list_display = ("name","location","station")
    list_filter = ("station",)
    search_fields = ("name__startswith",)
@admin.register(Crime)
class CrimeAdmin(admin.ModelAdmin):
    list_display = ("classification","correct","validated","date","image")
    search_fields = ("classification__startswith",)
# admin.site.register(Station)
# admin.site.register(CCTV)
# admin.site.register(Crime)