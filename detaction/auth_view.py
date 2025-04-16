from django.contrib.auth import authenticate, login,get_user_model,logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from asgiref.sync import sync_to_async


User = get_user_model()

def signup(request):
    if request.method == "POST":
        first_name = request.POST["first_name"]
        last_name = request.POST["last_name"]
        email = request.POST["email"]
        password = request.POST["password"]
        confirm_password = request.POST["confirm_password"]

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect("signup")

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect("signup")

        user = User.objects.create_user(email=email, first_name=first_name, last_name=last_name, password=password)
        messages.success(request, "Account created successfully. Please login.")
        return redirect("login")

    return render(request, "signup.html")

def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        
        print("Authenticating user:", email)  
        
        user = authenticate(request, username=email, password=password)  
        if user is not None:
            print("User authenticated:", user)  
            login(request, user)  
            return redirect("home")  
        else:
            messages.error(request, "Invalid email or password")
            return redirect("login")

    return render(request, "login.html")

@login_required
def logout_view(request):
    logout(request)  
    return redirect("login")  
