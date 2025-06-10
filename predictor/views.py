from django.shortcuts import render,redirect
from sklearn import linear_model
from .forms import Parameters
from .regressor import LogisticRegression ,load_data
import pandas as pd
import numpy as np
from . import regressor
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression 
from django.contrib import messages
from django.contrib.auth.models import User,auth
from django.contrib.auth import authenticate,login 
from django.contrib.auth.decorators import login_required
from .models import HeartData,DoctorHospital
from django.core.mail import send_mail
from django.conf import settings
# Create your views here.
from .regressor import LogisticRegression, load_data
from sklearn.model_selection import train_test_split
import numpy as np

def quickpredict(request):  # Predicts results after form is submitted; works only if user is not authenticated
    if request.method == 'POST':
        form = Parameters(request.POST)
        if form.is_valid():
            # Retrieve form data
            age = form.cleaned_data['age']
            sex = form.cleaned_data['sex']
            cp = form.cleaned_data['cp']
            trestbps = form.cleaned_data['trestbps']
            chol = form.cleaned_data['chol']
            fbs = form.cleaned_data['fbs']
            restcg = form.cleaned_data['restcg']
            thalach = form.cleaned_data['thalach']
            exang = form.cleaned_data['exang']
            oldpeak = form.cleaned_data['oldpeak']
            slope = form.cleaned_data['slope']
            ca = form.cleaned_data['ca']
            thal = form.cleaned_data['thal']
            
            # Load data and train model
            X, Y = load_data()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
            model = LogisticRegression(learning_rate=0.0001, iterations=1000)
            model.fit(X_train, Y_train)
            
            # Prediction
            input_data = np.array([age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            output, output_prob = model.predict(input_data)
            output1 = output_prob[0]
            
            return render(request, 'output.html', {'output1': output1})
        else:
            return redirect('/')

    else:
        form = Parameters()
        return render(request, 'quickpredict.html', {'form': form})


def index(request): #Directs the user to home page . Different for authenticated and non authenticated users.
    if request.user.is_authenticated:
        if request.method=='POST':
    
            form=Parameters(request.POST)
            if form.is_valid():
                age=form.cleaned_data['age']
                sex=form.cleaned_data['sex']
                cp=form.cleaned_data['cp']
                trestbps=form.cleaned_data['trestbps']
                chol=form.cleaned_data['chol']
                fbs=form.cleaned_data['fbs']
                restcg=form.cleaned_data['restcg']
                thalach=form.cleaned_data['thalach']
                exang=form.cleaned_data['exang']
                oldpeak=form.cleaned_data['oldpeak']
                slope=form.cleaned_data['slope']
                ca=form.cleaned_data['ca']
                thal=form.cleaned_data['thal']


                X,Y= load_data()
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )
                #model = LogitRegression(learning_rate=0.01 , iterations=1000)
                model = LogisticRegression(learning_rate=0.0001 , iterations=1000)
                model.fit(X_train, Y_train)
                output , output1 = model.predict(np.array([age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1))
                danger = 'high' if output == 1 else 'low'
                output1=output1[0]
                saved_data = HeartData(age=age ,  # Saving to database
                sex = sex,
                cp = cp,
                trestbps = trestbps,
                chol = chol,
                fbs = fbs,
                restcg = restcg , 
                thalach = thalach , 
                exang = exang,
                oldpeak = oldpeak,
                slope = slope,
                ca = ca,
                thal = thal,
                owner = request.user,
                probability = output1
                )  #Saved the authenticated users data in the database.
                saved_data.save()
                return render(request , 'output2.html',{'output1':output1 , 'danger':danger})


        form = Parameters()
        return render(request , 'user.html', {'form':form})
    return render(request , 'index.html')



def record(request): #Diplays previous record of authenticated user
    if request.user.is_authenticated:
        record_data = HeartData.objects.filter(owner = request.user) #Filter only those data whose owner is the logged in user
        return render(request , 'record.html' , {'record_data':record_data})
    return redirect('/')

def heartdetail(request): # Displays previous results in detail
    if request.user.is_authenticated:
        record_data = HeartData.objects.filter(owner = request.user)
        return render(request , 'heartdetail.html' , {'record_data':record_data})
    return redirect('/')

def symptoms(request): # Diplays list of symptoms of heart disease
    if request.user.is_authenticated:
        record_data = HeartData.objects.all()
        return render(request , 'symptoms.html')
    return redirect('/')

def prevention(request): # Diplays list of prevention of heart disease
    if request.user.is_authenticated:
        return render(request , 'prevention.html')
    return render('/')

def doctorhospital(request): # Displays list of doctors and hospitals for user
    if request.user.is_authenticated:
        datas = DoctorHospital.objects.all()
        return render(request , 'doctorshospitals.html',{'datas':datas})
    return render('/')







def about(request): #Displays about us page.
    return render(request , 'about.html')

# Login and Logout





def signin(request):
    if request.user.is_authenticated:
        if request.user.is_staff:
            return redirect('admin:index')
        return redirect('index')

    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # Clear any existing sessions first
        request.session.flush()

        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            if user.is_staff:
                messages.warning(request, 'Admin users must use the admin login page')
                return redirect('admin:login')
            auth.login(request, user)
            return redirect('index')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('signin')
    
    return render(request, 'signin.html')


def signup(request): #For the user to resister or sign up.

    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']

        
        if User.objects.filter(username=username).exists():
            messages.info(request,'Username taken')
            return redirect('signup')
        elif User.objects.filter(email=email).exists():
            messages.info(request,'Email taken')
            return redirect('signup')
        else:
            user = User.objects.create_user(username=username, password=password,email=email,first_name = first_name,last_name=last_name)
            
            user.save()
            
            messages.success(request,f"User {username} created!")
            return redirect('signin')
        #return redirect('/')
    else:   
        return render(request,'signup.html')


def signout(request):
    # Store the is_staff status before logout
    was_staff = request.user.is_staff
    
    # Clear all session data
    request.session.flush()
    auth.logout(request)
    
    # Redirect based on user type
    if was_staff:
        return redirect('admin:login')
    return redirect('index')


# End login