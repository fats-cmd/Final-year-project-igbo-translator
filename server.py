from flask import Flask, render_template, request, url_for, redirect
from translate import Translator

app = Flask(__name__)

translator = Translator(to_lang='ig')


@app.route('/')
@app.route('/index', methods=['POST'])
def home():
    if request.method == 'POST':
        untranslated = request.form.get('untranslated')
        the_translated = translator.translate(untranslated)
        return render_template('index.html',the_translated=the_translated, untranslated=untranslated)
    else:
        return render_template('index.html')

# @app.route('/index', methods=['POST','GET'])
# def index():
#     return render_template('index.html', )