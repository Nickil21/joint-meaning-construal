import glob
import os
import shutil
import csv
from flask import Flask, render_template, request, redirect, url_for, flash, session, escape
from werkzeug.utils import secure_filename
from source.video_splitter import split_by_seconds
import pandas as pd
from source import inference


class Pager(object):
    def __init__(self, count):
        self.count = count
        self.current = 0

    @property
    def next(self):
        n = self.current + 1
        if n > self.count-1:
            n -= self.count
        return n

    @property
    def prev(self):
        n = self.current - 1
        if n < 0 :
            n += self.count
        return n


def read_table(url):
    """Return a list of dict"""
    # r = requests.get(url)
    with open(url) as f:
        return [row for row in csv.DictReader(f.readlines())]


UPLOAD_FOLDER = 'static/uploads/'
APPNAME = "Hand Gesture Classification"
STATIC_FOLDER = 'static/'
TABLE_FILE = "static/hand_gestures_predicted.csv"

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.secret_key = "secret key"
app.config.update(APPNAME=APPNAME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # Max 5 MB file size


@app.route('/upload', methods=['GET'])
def upload_form():
    if 'username' in session:
        flash('Hey, {}!'.format(escape(session['username'])))
    else:
        flash('You are not signed in!')
    # delete uploaded files
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading', 'error')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        if not filename.endswith(".mp4"):
            flash('Please input a mp4 type video file!', 'error')
            return redirect(request.url)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and the hand gestures displayed below', 'success')
        # hand_gestures = evaluate(test_video=os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('index.html', filename=filename)


@app.route('/display/<filename>')
def display_video(filename):
    print('display_video filename: ' + filename)
    chunk = "chunk_" + filename.split("_")[1]
    split_by_seconds(filename="static/uploads/"+ filename, segment_path=None, split_length=0.5, chunk="static/uploads/" + chunk, vcodec="libx264", extra="-vf 'scale=1280:720' -threads 32")
    session['chunk'] = chunk
    l = []
    for f in glob.glob("static/uploads/{}/*".format(session['chunk'])):
        d = inference.evaluate(f)
        d['name'] = f.split("/")[-1]
        l.append(d)
    df = pd.DataFrame(l)
    df.sort_values('name', inplace=True)
    df.to_csv("static/hand_gestures_predicted.csv", index=False)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/<int:ind>/')
def image_view(ind=None):
    table = read_table(TABLE_FILE)
    pager = Pager(len(table))
    if ind >= pager.count:
        return render_template("404.html"), 404
    else:
        pager.current = ind
        return render_template('videoview.html', index=ind, pager=pager, data=table[ind])


@app.route('/goto', methods=['POST', 'GET'])
def goto():
    return redirect('/' + request.form['index'])


@app.route('/', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('upload_form'))
    return '''
        <form action="" method="post">
            <p>Username <input name="username" required=true>
            <p><button>Sign in</button></p>
        </form>
    '''


@app.route('/sign_out')
def sign_out():
    session.clear()
    return redirect(url_for('sign_in'))


if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True, host="localhost")
