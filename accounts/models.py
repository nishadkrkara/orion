from django.db import models
from django.contrib.auth.models import Group
from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from branch.models import Branch
import random

# function for generate random string


def generate_user_id():
    user_id =  "".join([random.choice('ABCDEFGHIJKLMNPQRSTUVXYZ123456789') for i in range(8)])
    return user_id
    
class UserAccountManager(BaseUserManager):

    use_in_migrate=True

    def _create_user(self,mobile,email,password,**extra_fields):
        if not email:
            raise ValueError("Email Address Must Be Provided")
        if not mobile:
            raise ValueError("Contact Number Must be provided")
        if not password:
            raise ValueError("Password Must be provided")
        
        email=self.normalize_email(email)

        user=self.model(mobile=mobile,
                        email=email,
                        **extra_fields)

        user.set_password(password)

        user.save(using=self._db)

        return user

    def create_superuser(self,email=None, mobile=None,password=None,**extra_fields):
        
        extra_fields["is_superuser"]=True
        
        extra_fields["is_staff"]=True
        
        return self._create_user( mobile,email,password,**extra_fields)





class Role(Group):

    role_choice = [('ADMIN NISHAD','ADMIN'),('SALES PERSON','SALES PERSON')]           
    role_name = models.CharField(max_length=65,choices = role_choice)
    role_description = models.CharField(max_length=165, null=True)
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)




class CustomUser(AbstractBaseUser, PermissionsMixin):

    REQUIRED_FIELDS=["mobile"]
    USERNAME_FIELD="email"

    first_name = models.CharField(max_length=255, null=True)
    role = models.ForeignKey(Role,on_delete=models.CASCADE,blank=True, null=True) 
    last_name = models.CharField(max_length=255,null=True)
    email = models.EmailField(unique=True)
    mobile = models.CharField(max_length=15)
    custom_id = models.CharField(max_length=255, unique=True,default=generate_user_id) # unique id of each user
    is_superuser = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)
    is_system_admin = models.BooleanField(default=False)
    is_user = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True, null=True)
    start_date = models.DateField(blank=True, null=True)
    end_date = models.DateField(blank=True, null=True)
    gender = models.CharField(max_length=10, blank=True, null=True, default=None)
    user_pic = models.ImageField(upload_to='user_images/', blank=True, null=True)
    date_of_birth = models.IntegerField(blank=True, null=True)
    blood_type = models.IntegerField(blank=True, null=True)
    created_by = models.IntegerField(blank = True,null = True)
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)
    login_expiry_time = models.IntegerField(blank=True, null=True)
    pincode = models.CharField(max_length=6, blank=True, null=True)
    branch = models.ForeignKey(Branch, null=True, on_delete=models.CASCADE)
    objects = UserAccountManager()

    

    class Meta:
        db_table = 'orion_custom_user'

    def __str__(self):
        return self.email




