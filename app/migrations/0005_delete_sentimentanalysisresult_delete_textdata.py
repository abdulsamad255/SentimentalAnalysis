# Generated by Django 5.0.4 on 2024-05-14 19:03

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0004_sentimentanalysisresult'),
    ]

    operations = [
        migrations.DeleteModel(
            name='SentimentAnalysisResult',
        ),
        migrations.DeleteModel(
            name='TextData',
        ),
    ]
