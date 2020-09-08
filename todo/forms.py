from django import forms

class TodoForm(forms.Form):
    text = forms.CharField(max_length=40,
        widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder' : 'Enter todo e.g. Delete junk files', 'aria-label' : 'Todo', 'aria-describedby' : 'add-btn'}))
    faces = forms.FileField()
    comment = forms.CharField(max_length=40,
        widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder' : 'Enter todo comment', 'aria-label' : 'Todo', 'aria-describedby' : 'add-btn'}))
    vlink = forms.CharField(max_length=40,
        widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder' : 'Enter video link', 'aria-label' : 'Todo', 'aria-describedby' : 'add-btn'}))
    lat = forms.CharField(max_length=40,
        widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder' : 'Enter latitude', 'aria-label' : 'Todo', 'aria-describedby' : 'add-btn'}))
    lon = forms.CharField(max_length=40,
        widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder' : 'Enter longitude', 'aria-label' : 'Todo', 'aria-describedby' : 'add-btn'}))
    zom = forms.CharField(max_length=40,
        widget=forms.TextInput(
            attrs={'class' : 'form-control', 'placeholder' : 'Enter Zoom', 'aria-label' : 'Todo', 'aria-describedby' : 'add-btn'}))
