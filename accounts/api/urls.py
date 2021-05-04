from . import views
from django.urls import path

urlpatterns = [path('user-create/',views.UserAccountCreateAPI.as_view()),
              path('user-login/',views.UserLogin.as_view()),
              path('me/',views.UserLoginInfo.as_view())
]