# Generated by Django 2.2 on 2019-04-18 20:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('face_detector', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='author',
            field=models.CharField(max_length=200),
        ),
    ]
