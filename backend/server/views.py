from django.shortcuts import render
from django.http import HttpResponse
from NeuralNet import CNN

def index(request):
    x = CNN.createCNN()
    return HttpResponse("Hello, you are at the python server")

# Create your views here.
