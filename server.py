from flask import Flask, render_template, request, url_for, redirect, session, jsonify

from translate import Translator

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session







@app.route('/')
@app.route('/index', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        untranslated = request.form.get('untranslated')
        from_lang = request.form.get('from_lang', 'en')
        to_lang = request.form.get('to_lang', 'ig')
        error = None
        if from_lang == to_lang:
            the_translated = untranslated
        else:
            try:
                translator = Translator(from_lang=from_lang, to_lang=to_lang)
                the_translated = translator.translate(untranslated)
            except Exception as e:
                the_translated = ''
                error = f"Error: {str(e)}"

        # Save to history in session
        if 'history' not in session:
            session['history'] = []
        if untranslated and the_translated:
            session['history'] = ([{
                'from': from_lang,
                'to': to_lang,
                'input': untranslated,
                'output': the_translated
            }] + session['history'])[:10]  # Keep last 10
        session.modified = True

        # Favorites
        favorites = session.get('favorites', [])

        return render_template('index.html', the_translated=the_translated, untranslated=untranslated, error=error, history=session['history'], favorites=favorites)
    else:
        # GET: show history/favorites
        history = session.get('history', [])
        favorites = session.get('favorites', [])
        return render_template('index.html', history=history, favorites=favorites)

# Add to favorites endpoint
@app.route('/favorite', methods=['POST'])
def favorite():
    input_text = request.form.get('input')
    output_text = request.form.get('output')
    from_lang = request.form.get('from')
    to_lang = request.form.get('to')
    if not input_text or not output_text:
        return jsonify({'success': False, 'msg': 'Missing data'}), 400
    fav = {'from': from_lang, 'to': to_lang, 'input': input_text, 'output': output_text}
    favorites = session.get('favorites', [])
    if fav not in favorites:
        favorites.insert(0, fav)
        session['favorites'] = favorites[:10]
        session.modified = True
    return jsonify({'success': True})

# Simple API endpoint
@app.route('/api/translate', methods=['POST'])
def api_translate():
    data = request.get_json(force=True)
    untranslated = data.get('untranslated')
    from_lang = data.get('from_lang', 'en')
    to_lang = data.get('to_lang', 'ig')
    if not untranslated:
        return jsonify({'error': 'No text provided'}), 400
    try:
        translator = Translator(from_lang=from_lang, to_lang=to_lang)
        the_translated = translator.translate(untranslated)
        return jsonify({'translated': the_translated})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/index', methods=['POST','GET'])
# def index():
#     return render_template('index.html', )