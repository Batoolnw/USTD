from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
#from django.http import StreamingHttpResponse

from . import views
#from . import emotions
#from . import snap



urlpatterns = [
    path('', views.index, name='index'),
    path('add', views.addTodo, name='add'),
    path('complete/<todo_id>', views.completeTodo, name='complete'),
    path('uncomplete/<todo_id>', views.uncompleteTodo, name='uncomplete'),
    path('deletecomplete', views.deleteCompleted, name='deletecomplete'),
    path('deleteall', views.deleteAll, name='deleteall'),
    path("stream",views.indexscreen,name="stream"),
    path("video",views.indexvideo,name="video"),
    #path("vlink",emotions,name="vlink"),
    path('videostream', views.dynamic_stream,name='videostream'),
    path('vg', views.vg, name='vg'),
    path('tf', views.tf, name='tf'),
    path('snap', views.snap, name='snap'),
    path('index2', views.index2, name='index2'),
]



if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
