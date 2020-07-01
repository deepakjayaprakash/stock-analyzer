from django.http import HttpResponse


def testAPI(request):
    return HttpResponse("This means its working")
