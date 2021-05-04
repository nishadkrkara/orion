from rest_framework.generics import GenericAPIView
from django.urls import path
from accounts.api.serializers import UserCreateSerializer,UserLoginSerializer,UserLoginInfoSerailizer
from django.http import JsonResponse
from rest_framework.permissions import AllowAny
from accounts.models import CustomUser



class UserAccountCreateAPI(GenericAPIView):

    serializer_class = UserCreateSerializer

    def post(self,request):

        serializer = self.get_serializer(data=request.data)
        if serilizer.is_valid():
            serializer.save()
            return JsonResponse({'code':201,'message':'User created Successfully'})
        return JsonResponse({'code':200,'message':serializer.errors})



class UserLogin(GenericAPIView):

    serializer_class = UserLoginSerializer
    permission_classes = [AllowAny,]

    def post(self,request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        return JsonResponse({
                'code':200,
                'message':'Successfully Logined',
                'response':serializer.data
            })
    

class UserLoginInfo(GenericAPIView):

    serializer_class = UserLoginInfoSerailizer

    def get(self,request):
        serializer = self.serializer_class(request.user,many=False)
        print(serializer.data)
        return JsonResponse({'code':200,'response':serializer.data})