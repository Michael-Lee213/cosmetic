from flask import Blueprint, render_template

# Create the Blueprint object (THIS WAS MISSING)
bp = Blueprint('main', __name__)  # 'main' is the name of the blueprint

@bp.route('/')
def index():
    return render_template('input.html')

