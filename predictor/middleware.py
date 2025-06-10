from django.conf import settings
from django.contrib.auth import logout
from django.shortcuts import redirect

class SessionSeparationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        is_admin_path = request.path.startswith('/admin/')
        
        # Set the appropriate session cookie name
        if is_admin_path:
            request.session.cookie_name = settings.ADMIN_SESSION_COOKIE_NAME
        else:
            request.session.cookie_name = settings.SESSION_COOKIE_NAME

        # Handle authenticated users
        if request.user.is_authenticated:
            # Admin users should only access admin paths
            if request.user.is_staff:
                if not is_admin_path:
                    logout(request)
                    return redirect('signin')
            # Regular users should not access admin paths
            else:
                if is_admin_path:
                    logout(request)
                    return redirect('admin:login')

        response = self.get_response(request)

        # Set appropriate cookie settings only if the cookie exists
        if is_admin_path and settings.ADMIN_SESSION_COOKIE_NAME in response.cookies:
            response.cookies[settings.ADMIN_SESSION_COOKIE_NAME].update({
                'max-age': settings.ADMIN_SESSION_COOKIE_AGE,
                'httponly': True,
                'samesite': 'Lax'
            })
        elif not is_admin_path and settings.SESSION_COOKIE_NAME in response.cookies:
            response.cookies[settings.SESSION_COOKIE_NAME].update({
                'max-age': settings.SESSION_COOKIE_AGE,
                'httponly': True,
                'samesite': 'Lax'
            })

        return response 