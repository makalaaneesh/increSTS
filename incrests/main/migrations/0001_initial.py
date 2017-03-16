# -*- coding: utf-8 -*-
# Generated by Django 1.10.6 on 2017-03-16 08:55
from __future__ import unicode_literals

from django.db import migrations, models
import main.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Process',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to=main.models.file_path)),
                ('date', models.DateTimeField(auto_now=True)),
                ('pid', models.CharField(blank=True, max_length=100, null=True)),
                ('status', models.BooleanField(default=False)),
            ],
        ),
    ]
