from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.views.decorators.http import require_POST

from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError
from django.views.decorators import gzip

#import os.path as osp
import time
import cv2
import matplotlib.cm as cm
import numpy as np
import torch.hub
import os
from .model import Model
from PIL import Image
from torchvision import transforms
from torchsummary import summary
from .grad_cam import BackPropagation, GradCAM,GuidedBackPropagation
from datetime import datetime

from .models import Todo
from .forms import TodoForm
from .snap import mainsnap
#######
from keras.models import load_model
from statistics import mode
from .utils.datasets import get_labels
from .utils.inference import detect_faces
from .utils.inference import draw_text
from .utils.inference import draw_bounding_box
from .utils.inference import apply_offsets
from .utils.inference import load_detection_model
from .utils.preprocessor import preprocess_input

#######


def index2(request):
    todo_list = Todo.objects.order_by('-id')

    form = TodoForm()

    context = {'todo_list' : todo_list, 'form' : form}

    return render(request, 'todo/index.html', context)

#####
#######


def index(request):
    todo_list = Todo.objects.order_by('-id')



# 	os.mkdir(os.path.join('', 'Media-Snap-Map'))

    dirname = datetime.now().strftime('%Y.%m.%d.%H') #2010.08.09.12.08.45
    if not os.path.isdir('Media-Snap-Map/' + dirname): (os.mkdir(os.path.join('Media-Snap-Map/', dirname)))
    #mainsnap()

    form = TodoForm()

    context = {'todo_list' : todo_list, 'form' : form}

    return render(request, 'todo/newindex.html', context)

#####

def tf(request):

    todo_list = Todo.objects.order_by('-id')

    form = TodoForm()

    context = {'todo_list' : todo_list, 'form' : form}

    return render(request, 'todo/tf.html', context)

#####
#####
def snap(request):
    from . import snap

    snaps = snap.getSnaps('24.83893115615588','46.71576928913487')

    form = TodoForm()

    context = {'form' : form, 'snaps' : snaps}

    return render(request, 'todo/snap.html', context)

#####
#####
def vg(request):

    todo_list = Todo.objects.order_by('-id')

    form = TodoForm()

    context = {'todo_list' : todo_list, 'form' : form}

    return render(request, 'todo/force.html', context)

#####
@require_POST
def addTodo(request):
    #form = TodoForm(request.POST, request.FILES)

    #if form.is_valid():
    uploaded_file = request.FILES['faces']
    fs = FileSystemStorage()
    fs.save(uploaded_file.name, uploaded_file)

    faceCascade = cv2.CascadeClassifier('/home/happyfaces/django_todo_app/todo/haarcascade_frontalface_default.xml')
    shape = (48,48)
    classes = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprised',
        'Neutral'
    ]


    def preprocess(image_path):
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        image = cv2.imread(image_path)
        faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print('no face found')
            face = cv2.resize(image, shape)
        else:
            (x, y, w, h) = faces[0]
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, shape)

        img = Image.fromarray(face).convert('L')
        inputs = transform_test(img)
        return inputs, face


    def get_gradient_image(gradient):
        gradient = gradient.cpu().numpy().transpose(1, 2, 0)
        gradient -= gradient.min()
        gradient /= gradient.max()
        gradient *= 255.0
        return np.uint8(gradient)


    def get_gradcam_image(gcam, raw_image, paper_cmap=False):
        gcam = gcam.cpu().numpy()
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        if paper_cmap:
            alpha = gcam[..., None]
            gcam = alpha * cmap + (1 - alpha) * raw_image
        else:
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        return np.uint8(gcam)


    def guided_backprop(images, model_name):

        for i, image in enumerate(images):
            target, raw_image = preprocess(image['path'])
            image['image'] = target
            image['raw_image'] = raw_image

        net = Model(num_classes=len(classes))
        checkpoint = torch.load(os.path.join('', model_name), map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        summary(net, (1, shape[0], shape[1]))

        #result_images = []
        for index, image in enumerate(images):
            img = torch.stack([image['image']])
            bp = BackPropagation(model=net)
            probs, ids = bp.forward(img)
            gcam = GradCAM(model=net)
            _ = gcam.forward(img)

            gbp = GuidedBackPropagation(model=net)
            _ = gbp.forward(img)

            # Guided Backpropagation
            actual_emotion = ids[:,0]
            gbp.backward(ids=actual_emotion.reshape(1,1))
            gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=actual_emotion.reshape(1,1))
            regions = gcam.generate(target_layer='last_conv')

            # Get Images
            label_image = np.zeros((shape[0],65, 3), np.uint8)
        return classes[actual_emotion.data]

    label = guided_backprop(
                images=[
                    {'path': '/home/happyfaces/django_todo_app/media/' + uploaded_file.name},
                ],
                model_name='/home/happyfaces/django_todo_app/todo/private_model_233_66.t7'
            )

    new_todo = Todo(imglnk=uploaded_file.name, comment=label)
    new_todo.save()



    return redirect('index')
    #print (request.POST['text'])
##################
def showvideo(request):

    lastvideo= Video.objects.last()

    videofile= lastvideo.videofile


    form= VideoForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        form.save()


    context= {'videofile': videofile,
              'form': form
              }


    return render(request, 'videos.html', context)

#################
def completeTodo(request, todo_id):
    todo = Todo.objects.get(pk=todo_id)
    todo.complete = True
    todo.save()

    return redirect('index')

def uncompleteTodo(request, todo_id):
    todo = Todo.objects.get(pk=todo_id)
    todo.complete = False
    todo.save()

    return redirect('index')

def deleteCompleted(request):
    Todo.objects.filter(complete__exact=True).delete()

    return redirect('index')

def deleteAll(request):
    Todo.objects.all().delete()

    return redirect('index')



def get_frame():
    camera =cv2.VideoCapture(0)
    while True:
        _, img = camera.read()
        imgencode=cv2.imencode('.jpg',img)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(camera)

def indexscreen(request):
    template = "todo/screens.html"
    return render(request,template)

def testv(request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    template = "todo/test.html"
    return render(request,template)

def indexvideo(request):
    template = "todo/video.html"
    return render(request,template)

def vlink(request):
    USE_WEBCAM = True # If false, loads video file source

    # parameters for loading data and images
    emotion_model_path = '/home/happyfaces/django_todo_app/todo/models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_cascade = cv2.CascadeClassifier('/home/happyfaces/django_todo_app/todo/models/haarcascade_frontalface_default.xml')
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []

    # starting video streaming

    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)

    # Select video or webcam feed
    cap = None
    if (USE_WEBCAM == True):
        cap = cv2.VideoCapture(0) # Webcam source
    else:
        cap = cv2.VideoCapture('./demo/dinner.mp4') # Video file source

    while cap.isOpened(): # True:
        ret, bgr_image = cap.read()

        #bgr_image = video_capture.read()[1]

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
    			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            color = color.astype(int)
            color = color.tolist()

            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, emotion_mode,
                      color, 0, -45, 1, 1)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
       # cv2.imshow('window_frame', bgr_image)
       # if cv2.waitKey(1) & 0xFF == ord('q'):
       #     break
    	#frame_flip = cv2.flip(bgr_image,1)
		#ret, jpeg = cv2.imencode('.jpg', bgr_image)
		#vid = jpeg.tobytes()
	#return StreamingHttpResponse(gen(vid)),
					#content_type='multipart/x-mixed-replace; boundary=frame')
    #cap.release()
    #cv2.destroyAllWindows()

#####################

    faceCascade = cv2.CascadeClassifier('/home/happyfaces/django_todo_app/todo/haarcascade_frontalface_default.xml')
    shape = (48,48)
    classes = [
        'Angry',
        'Disgust',
        'Fear',
        'Happy',
        'Sad',
        'Surprised',
        'Neutral'
    ]


    def preprocess(image_path):
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        image = cv2.imread(image_path)
        faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            print('no face found')
            face = cv2.resize(image, shape)
        else:
            (x, y, w, h) = faces[0]
            face = image[y:y + h, x:x + w]
            face = cv2.resize(face, shape)

        img = Image.fromarray(face).convert('L')
        inputs = transform_test(img)
        return inputs, face


    def get_gradient_image(gradient):
        gradient = gradient.cpu().numpy().transpose(1, 2, 0)
        gradient -= gradient.min()
        gradient /= gradient.max()
        gradient *= 255.0
        return np.uint8(gradient)


    def get_gradcam_image(gcam, raw_image, paper_cmap=False):
        gcam = gcam.cpu().numpy()
        cmap = cm.jet_r(gcam)[..., :3] * 255.0
        if paper_cmap:
            alpha = gcam[..., None]
            gcam = alpha * cmap + (1 - alpha) * raw_image
        else:
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        return np.uint8(gcam)


    def guided_backprop(images, model_name):

        for i, image in enumerate(images):
            target, raw_image = preprocess(image['path'])
            image['image'] = target
            image['raw_image'] = raw_image

        net = Model(num_classes=len(classes))
        checkpoint = torch.load(os.path.join('', model_name), map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        net.eval()
        summary(net, (1, shape[0], shape[1]))

        #result_images = []
        for index, image in enumerate(images):
            img = torch.stack([image['image']])
            bp = BackPropagation(model=net)
            probs, ids = bp.forward(img)
            gcam = GradCAM(model=net)
            _ = gcam.forward(img)

            gbp = GuidedBackPropagation(model=net)
            _ = gbp.forward(img)

            # Guided Backpropagation
            actual_emotion = ids[:,0]
            gbp.backward(ids=actual_emotion.reshape(1,1))
            gradients = gbp.generate()

            # Grad-CAM
            gcam.backward(ids=actual_emotion.reshape(1,1))
            regions = gcam.generate(target_layer='last_conv')

            # Get Images
            label_image = np.zeros((shape[0],65, 3), np.uint8)
        return classes[actual_emotion.data]

    label = guided_backprop(
                images=[
                    {'path': '/home/happyfaces/django_todo_app/media/' + uploaded_file.name},
                ],
                model_name='/home/happyfaces/django_todo_app/todo/private_model_233_66.t7'
            )

    new_todo = Todo(imglnk=uploaded_file.name, comment=label)
    new_todo.save()



    return redirect('index')
    #print (request.POST['text'])


@gzip.gzip_page
def dynamic_stream(request,stream_path="video"):
    try :
        return StreamingHttpResponse(get_frame(),content_type="multipart/x-mixed-replace;boundary=frame")
    except :
        return "error"