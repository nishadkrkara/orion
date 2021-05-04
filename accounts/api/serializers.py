from rest_framework import serializers
from accounts.models import CustomUser
from django.contrib.auth import authenticate
from django.contrib.auth.models import update_last_login
from rest_framework_jwt.settings import api_settings


JWT_PAYLOAD_HANDLER = api_settings.JWT_PAYLOAD_HANDLER
JWT_ENCODE_HANDLER = api_settings.JWT_ENCODE_HANDLER

class UserLoginSerializer(serializers.Serializer):
    email = serializers.CharField(max_length=255)
    role = serializers.CharField(max_length=40,read_only=True)
    password = serializers.CharField(max_length=128, write_only=True)
    token = serializers.CharField(max_length=255,read_only=True)

    def validate(self,data):
        email = data.get('email')
        print('Email  ',email)
        password = data.get('password')
        print("password  ",password)
        user = authenticate(email=email, password=password)
        print("user:  ",user)
        if user is None:
            raise serializers.ValidationError(
                'A user with email and password not found'
            )
        
        try:
            payload = JWT_PAYLOAD_HANDLER(user)
            print("payload  ",payload)
            jwt_token = JWT_ENCODE_HANDLER(payload)
            print("jwt token  ",jwt_token)
            update_last_login(None,user)
        
        except User.DoesNotExist:

            raise serializers.ValidationError(
                'User with given email and password does not exists'
            )
    

        role = user.role.role_name if user.role else "role not assigned"
        print("role   ",role)
        return {
            'email':user.email,
            'role':role,
            'token':jwt_token
        }


class UserCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['email',
                'mobile','branch',
                'first_name','role',
                'password'] 


class UserLoginInfoSerailizer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ['email','branch','mobile']