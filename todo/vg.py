from django.shortcuts import render
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_POST

from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError

from .models import Todo
from .forms import TodoForm

render(request,'force.html')