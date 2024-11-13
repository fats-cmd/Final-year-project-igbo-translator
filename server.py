from flask import Flask, render_template, request
from translate import Translator

app = Flask(__name__)

translator = Translator(to_lang='ig')



# @app.route('/')
# def index():
#     return render_template('index.html')


@app.route('/index', methods=['POST'])
def index():
    untranslated = request.form.get('untranslated')
    the_translated = translator.translate(untranslated)
    return render_template('index.html', the_translated=the_translated)

# @app.route('/<string:dynamic>')
# def random_page(dynamic):
#     return 'Looks like the page you are looking for does not exist try " /index " to get to the translator!'
