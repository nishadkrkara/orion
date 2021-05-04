from django.db import models

# Create your models here.

class Branch(models.Model):
    
    branch_name = models.CharField(max_length=191, unique=True)
    city = models.CharField(max_length=191, null=True)
    state = models.CharField(max_length=191, null=True)
    country = models.CharField(max_length=191, null=True)
    created_at = models.DateField(auto_now_add=True)
    updated_at = models.DateField(auto_now=True)

    def __str__(self):
        return self.branch_name

    class Meta:
        db_table = 'orion_branch'
        ordering = ["-id"]
