from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
!pip install mtcnn
from mtcnn.mtcnn import MTCNN
!pip install git+https://github.com/raghagra/keras-vggface.git
!pip install keras_applications
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
class detector:
  import easyocr
  def __init__(self, lc):
    import easyocr
    import re
    self.reader = easyocr.Reader(['hi','en'])
    self.res={}
    self.res['adhar_card']=''
    self.res['name']=' '
  
  #face comprison function STARTS HERE  
  def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array
    
  def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat
  
  def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
	else:
		print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
    return score
  
  def match_proba(idpic,selfiepic):
    filenames=[idpic,selfiepic]
    embeddings = get_embeddings(filenames)
    
    res=is_match(embeddings[0], embeddings[1])
    return res
  #face comparison ends here
  def detection(self,file_path):
    import re
    result = self.reader.readtext(file_path,face_path)
    for i in result:
      temp=re.findall(r'[^a-zA-Z ]+',i[1])
      if len(temp)==0:
        self.res['name']=self.res['name']+i[1]
      if (len(re.findall(r'\d{2}/\d{2}/\d{4}',i[1]))>0):
        self.res['dob']=re.findall(r'\d{2}/\d{2}/\d{4}',i[1])[0]
        temp=self.res['dob'].split('/')
        self.res['dob']=str(int(temp[0]))+'/'+str(int(temp[1]))+'/'+str(int(temp[2]))
      if len(re.findall(r'[^0-9 ]+',i[1]))==0:
        self.res['adhar_card']=self.res['adhar_card']+i[1]
      if (i[1].find('male')>=0)|(i[1].find('MALE')>=0):
        self.res['gender']='male'
      if (i[1].find('female')>=0)|(i[1].find('FEMALE')>=0):
        self.res['gender']='female'
      self.res['name']=self.res['name'].replace('Government','').replace('of','').replace('India','').replace('VID','')
      self.res['face_comparison']=match_proba(id_pic_path,selfie_pic_path)
    return self.res

  