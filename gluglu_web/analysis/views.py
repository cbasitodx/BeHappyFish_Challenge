from django.shortcuts import render
from django.core.files.storage import default_storage
from main.test import analyze_image


def analyze_image(request):
    if request.method == "POST":
        image = request.FILES.get("image")
        weight = request.POST.get("weight") == "on"
        eyes = request.POST.get("eyes") == "on"
        shap = request.POST.get("shap") == "on"

        if image:
            path = default_storage.save(f"uploads/{image.name}", image)
            result = analyze_image(path, weight, eyes, shap)

            weight = result.get("weight", None)
            eyes = result.get("eyes", None)
            shap_path = result.get("shap", None)
            bb_path = result.get("bb_path", None)

            return render(request, "analysis/results.html", {
                "weight": weight,
                "eyes": eyes,
                "shap_path": shap_path,
                "bb_path": bb_path
            })

    return render(request, "analysis/index.html")
