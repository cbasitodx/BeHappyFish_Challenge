from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from django.templatetags.static import static

from main import main

import os


def analyze_image(request):
    if request.method == "POST":
        image = request.FILES.get("image")
        weight = request.POST.get("weight") == "on"
        eyes = request.POST.get("eyes") == "on"
        shap = request.POST.get("shap") == "on"

        if image:
            path = default_storage.save(f"uploads/{image.name}", image)
            bb_path = "./static/results/yolo.jpg"
            shap_path = "./static/results/shap.jpg"
            numbers_path = "./static/results/"
            result = main(path, bb_path, shap_path, numbers_path, eyes, weight, shap)

            weight = result.get("weight_classification", None)
            eyes = result.get("eye_classification", None)
            number_image = "/results/predict/" + image.name

            default_storage.delete(f"uploads/{image.name}")

            print(number_image)
            print(weight)

            return render(request, "analysis/results.html", {
                "weight": weight,
                "eyes": eyes,
                "shap_path": shap_path,
                "bb_path": bb_path,
                "number_image": number_image
            })

    return render(request, "analysis/index.html")
