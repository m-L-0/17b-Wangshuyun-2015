from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
from recover import *
from PIL import Image
from multiprocessing import Pool
import os

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        path = os.path.join(os.path.dirname(__file__), 'static/download',secure_filename(f.filename)) #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(path)
        im = Image.open(path)
        im = im.resize((51,39))
        im = np.array(im.convert('L'))
        im.shape = 1,1989
        pool=Pool()
        apiout = pool.map(apirun,[im])
        predction = apiout[0]
        return render_template('index.html',predction=predction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
