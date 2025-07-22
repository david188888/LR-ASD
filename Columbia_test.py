import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, pdb, math, python_speech_features, json, uuid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from deepface import DeepFace
import tempfile

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from model.faceDetector.s3fd import S3FD
from ASD import ASD

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "Columbia ASD Evaluation")

parser.add_argument('--videoName',             type=str, default="col",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="colDataPath",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="weight/pretrain_AVA.model",   help='Path for the pretrained model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--evalCol',               dest='evalCol', action='store_true', help='Evaluate on Columbia dataset')
parser.add_argument('--colSavePath',           type=str, default="/colDataPath",  help='Path for inputs, tmps and outputs')

parser.add_argument('--generateJson',          dest='generateJson', action='store_true', help='Generate speaker labels JSON file')
parser.add_argument('--jsonOutputPath',        type=str, default="speaker_labels.json",  help='Output path for JSON file')
parser.add_argument('--speakerThreshold',      type=float, default=0.75, help='Similarity threshold for speaker clustering')

args = parser.parse_args()


if args.evalCol == True:
	# The process is: 1. download video and labels(I have modified the format of labels to make it easiler for using)
	# 	              2. extract audio, extract video frames
	#                 3. scend detection, face detection and face tracking
	#                 4. active speaker detection for the detected face clips
	#                 5. use iou to find the identity of each face clips, compute the F1 results
	# The step 1 to 3 will take some time (That is one-time process). It depends on your cpu and gpu speed. For reference, I used 1.5 hour
	# The step 4 and 5 need less than 10 minutes
	# Need about 20G space finally
	# ```
	args.videoName = 'col'
	args.videoFolder = args.colSavePath
	args.savePath = os.path.join(args.videoFolder, args.videoName)
	args.videoPath = os.path.join(args.videoFolder, args.videoName + '.mp4')
	args.duration = 0
	if os.path.isfile(args.videoPath) == False:  # Download video
		link = 'https://www.youtube.com/watch?v=6GzxbrO0DHM&t=2s'
		cmd = "youtube-dl -f best -o %s '%s'"%(args.videoPath, link)
		output = subprocess.call(cmd, shell=True, stdout=None)
	if os.path.isdir(args.videoFolder + '/col_labels') == False: # Download label
		link = "1Tto5JBt6NsEOLFRWzyZEeV6kCCddc6wv"
		cmd = "gdown --id %s -O %s"%(link, args.videoFolder + '/col_labels.tar.gz')
		subprocess.call(cmd, shell=True, stdout=None)
		cmd = "tar -xzvf %s -C %s"%(args.videoFolder + '/col_labels.tar.gz', args.videoFolder)
		subprocess.call(cmd, shell=True, stdout=None)
		os.remove(args.videoFolder + '/col_labels.tar.gz')	
else:
	args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
	args.savePath = os.path.join(args.videoFolder, args.videoName)

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.pyworkPath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
		  dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.pyworkPath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# CPU: extract mfcc
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	numpy.save(featuresPath, mfcc)

def evaluate_network(files, args):
	# GPU: active speaker detection by pretrained model
	s = ASD()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	# durationSet = {1,2,4,6} # To make the result more reliable
	durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		allScore = [] # Evaluation use model
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def extract_deepface_features_from_track(track_video_path, sample_frames=5):
	"""
	使用DeepFace从track视频中提取人脸特征
	Args:
		track_video_path: track视频文件路径
		sample_frames: 采样帧数
	Returns:
		averaged_embedding: 平均人脸特征向量
	"""
	try:
		# 检查视频文件是否存在
		if not os.path.exists(track_video_path):
			sys.stderr.write(f"视频文件不存在: {track_video_path}\n")
			return None
			
		video = cv2.VideoCapture(track_video_path)
		if not video.isOpened():
			sys.stderr.write(f"无法打开视频文件: {track_video_path}\n")
			return None
			
		frames = []
		total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
		
		# 均匀采样帧
		if total_frames > 0:
			step = max(1, total_frames // sample_frames)
			for i in range(0, total_frames, step):
				video.set(cv2.CAP_PROP_POS_FRAMES, i)
				ret, frame = video.read()
				if ret and frame is not None and frame.size > 0:
					frames.append(frame)
					if len(frames) >= sample_frames:
						break
		
		video.release()
		
		if not frames:
			sys.stderr.write(f"无法从视频中提取有效帧: {track_video_path}\n")
			return None
		
		# 使用DeepFace提取特征
		embeddings = []
		for idx, frame in enumerate(frames):
			tmp_file_path = None
			try:
				# 验证帧内容
				if frame is None or frame.size == 0:
					continue
					
				# 创建唯一的临时文件路径
				tmp_file_path = f"/tmp/deepface_frame_{uuid.uuid4().hex}.jpg"
				
				# 写入图片文件并验证
				write_success = cv2.imwrite(tmp_file_path, frame)
				if not write_success:
					sys.stderr.write(f"图片写入失败: frame {idx}\n")
					continue
					
				# 检查文件是否真的被创建且有内容
				if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
					sys.stderr.write(f"临时文件创建失败或为空: {tmp_file_path}\n")
					continue
				
				# 使用DeepFace提取特征 - 使用Facenet模型
				result = DeepFace.represent(
					img_path=tmp_file_path,
					model_name='Facenet',
					enforce_detection=False,  # 允许在检测失败时继续
					detector_backend='opencv'
				)
				
				if result and len(result) > 0:
					embedding = numpy.array(result[0]['embedding'])
					embeddings.append(embedding)
					sys.stderr.write(f"成功提取特征: frame {idx}, 特征维度: {len(embedding)}\n")
				else:
					sys.stderr.write(f"DeepFace返回空结果: frame {idx}\n")
				
			except Exception as e:
				sys.stderr.write(f"DeepFace特征提取失败 (frame {idx}): {str(e)}\n")
				continue
			finally:
				# 确保临时文件被清理
				if tmp_file_path and os.path.exists(tmp_file_path):
					try:
						os.remove(tmp_file_path)
					except:
						pass
		
		if embeddings:
			# 返回所有帧特征的平均值
			averaged_embedding = numpy.mean(embeddings, axis=0)
			sys.stderr.write(f"成功提取DeepFace特征: {len(embeddings)}/{len(frames)} 帧, 平均特征维度: {len(averaged_embedding)}\n")
			return averaged_embedding
		else:
			sys.stderr.write(f"所有帧的DeepFace特征提取都失败: {track_video_path}\n")
			return None
			
	except Exception as e:
		sys.stderr.write(f"视频处理失败: {track_video_path}, 错误: {str(e)}\n")
		return None

def evaluate_network_with_deepface_features(files, args):
	# GPU: active speaker detection with DeepFace features extraction
	s = ASD()
	s.loadParameters(args.pretrainModel)
	sys.stderr.write("Model %s loaded from previous state! \r\n"%args.pretrainModel)
	s.eval()
	allScores = []
	allFeatures = []  # Store DeepFace features for each track
	durationSet = {1,1,1,2,2,2,3,3,4,5,6}
	
	sys.stderr.write("开始使用DeepFace提取人脸特征...\n")
	
	for file in tqdm.tqdm(files, total = len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0]
		
		# 1. 计算ASD说话检测分数
		_, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
		videoFeature = []
		
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224,224))
				face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = numpy.array(videoFeature)
		
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * 25)),:,:]
		
		allScore = []
		
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		
		allScore = numpy.round((numpy.mean(numpy.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)
		
		# 2. 使用DeepFace提取人脸特征
		video_path = os.path.join(args.pycropPath, fileName + '.avi')
		deepface_feature = extract_deepface_features_from_track(video_path, sample_frames=8)
		
		if deepface_feature is not None:
			allFeatures.append(deepface_feature)
		else:
			# 如果DeepFace失败，使用零向量作为fallback
			sys.stderr.write(f"警告: Track {fileName} DeepFace特征提取失败，使用零向量\n")
			allFeatures.append(numpy.zeros(128))  # FaceNet特征维度为128
	
	sys.stderr.write(f"DeepFace特征提取完成，成功提取 {len([f for f in allFeatures if not numpy.allclose(f, 0)])} 个有效特征\n")
	return allScores, allFeatures

def cluster_speakers_by_deepface_features(features, threshold=0.75):
	"""
	基于DeepFace人脸特征向量进行说话人聚类
	使用改进的聚类算法，自动确定说话人数量
	"""
	if len(features) == 0:
		return []
	
	features = numpy.array(features)
	n_tracks = len(features)
	
	# 过滤掉零向量（DeepFace提取失败的track）
	valid_indices = []
	valid_features = []
	for i, feature in enumerate(features):
		if not numpy.allclose(feature, 0):
			valid_indices.append(i)
			valid_features.append(feature)
	
	if len(valid_features) == 0:
		sys.stderr.write("警告: 没有有效的DeepFace特征，使用默认聚类\n")
		return [{'speaker_id': 'Speaker_1', 'tracks': list(range(n_tracks)), 'avg_feature': numpy.zeros(128)}]
	
	valid_features = numpy.array(valid_features)
	sys.stderr.write(f"有效特征数: {len(valid_features)} / {n_tracks}\n")
	
	# 计算有效特征之间的相似度矩阵
	sim_matrix = cosine_similarity(valid_features)
	
	# 使用改进的贪心聚类算法，针对DeepFace特征优化
	best_n_clusters = 1
	best_score = -1
	best_labels = numpy.zeros(len(valid_features), dtype=int)
	
	# DeepFace特征通常区分度更好，使用更高的阈值
	thresholds_to_try = [0.5]
	
	for sim_threshold in thresholds_to_try:
		# 使用贪心聚类（只对有效特征进行）
		labels = numpy.full(len(valid_features), -1, dtype=int)
		cluster_id = 0
		
		for i in range(len(valid_features)):
			if labels[i] == -1:  # 尚未分配聚类
				# 创建新聚类
				labels[i] = cluster_id
				
				# 寻找与当前track相似的其他tracks
				for j in range(i + 1, len(valid_features)):
					if labels[j] == -1 and sim_matrix[i][j] > sim_threshold:
						labels[j] = cluster_id
				
				cluster_id += 1
		
		n_clusters = cluster_id
		
		# 计算聚类质量分数
		if n_clusters > 1 and n_clusters <= min(len(valid_features), 5):
			intra_cluster_sim = 0
			inter_cluster_sim = 0
			intra_count = 0
			inter_count = 0
			
			for i in range(len(valid_features)):
				for j in range(i + 1, len(valid_features)):
					if labels[i] == labels[j]:
						intra_cluster_sim += sim_matrix[i][j]
						intra_count += 1
					else:
						inter_cluster_sim += sim_matrix[i][j]
						inter_count += 1
			
			if intra_count > 0 and inter_count > 0:
				score = (intra_cluster_sim / intra_count) - (inter_cluster_sim / inter_count)
				
				# 偏好合理数量的聚类（2-4个）
				if 2 <= n_clusters <= 4:
					score += 0.1
				
				if score > best_score:
					best_score = score
					best_n_clusters = n_clusters
					best_labels = labels.copy()
	
	# 如果所有阈值都导致只有1个聚类，使用更激进的分割
	if best_n_clusters == 1 and len(valid_features) >= 2:
		# 找到相似度最低的几对，强制分成不同聚类
		min_similarities = []
		for i in range(len(valid_features)):
			for j in range(i + 1, len(valid_features)):
				min_similarities.append((sim_matrix[i][j], i, j))
		
		min_similarities.sort()  # 按相似度升序排列
		
		# 强制分成多个聚类（如果有足够的tracks）
		if len(valid_features) >= 3:
			best_labels = numpy.zeros(len(valid_features), dtype=int)
			used_tracks = set()
			
			# 选择相似度最低的一对作为前两个聚类中心
			if min_similarities:
				sim, i, j = min_similarities[0]
				best_labels[i] = 0
				best_labels[j] = 1
				used_tracks.add(i)
				used_tracks.add(j)
				
				# 找到与前两个聚类中心差异最大的track作为第三个中心
				max_min_sim = -1
				third_center = -1
				for k in range(len(valid_features)):
					if k not in used_tracks:
						min_sim_to_centers = min(sim_matrix[k][i], sim_matrix[k][j])
						if min_sim_to_centers > max_min_sim:
							max_min_sim = min_sim_to_centers
							third_center = k
				
				if third_center != -1:
					best_labels[third_center] = 2
					used_tracks.add(third_center)
			
			# 分配剩余的tracks到最相似的聚类
			for i in range(len(valid_features)):
				if i not in used_tracks:
					best_sim = -1
					best_cluster = 0
					for j in range(len(valid_features)):
						if j in used_tracks and sim_matrix[i][j] > best_sim:
							best_sim = sim_matrix[i][j]
							best_cluster = best_labels[j]
					best_labels[i] = best_cluster
			
			best_n_clusters = len(set(best_labels))
		else:
			# 如果只有2个有效特征，强制分成2个聚类
			best_labels = numpy.array([0, 1])
			best_n_clusters = 2
	
	labels = best_labels
	
	# 构建说话人聚类结果，将有效特征的聚类结果映射回原始track索引
	speakers = []
	for cluster_id in range(best_n_clusters):
		valid_track_indices = [i for i, label in enumerate(labels) if label == cluster_id]
		if valid_track_indices:
			# 将有效特征索引转换为原始track索引
			original_track_indices = [valid_indices[i] for i in valid_track_indices]
			cluster_features = valid_features[valid_track_indices]
			avg_feature = numpy.mean(cluster_features, axis=0)
			
			speakers.append({
				'speaker_id': f"Speaker_{cluster_id + 1}",
				'tracks': original_track_indices,
				'avg_feature': avg_feature
			})
	
	# 将DeepFace提取失败的tracks分配给最近的聚类（如果有的话）
	failed_tracks = [i for i in range(n_tracks) if i not in valid_indices]
	if failed_tracks and speakers:
		# 将失败的tracks分配给第一个聚类
		speakers[0]['tracks'].extend(failed_tracks)
		sys.stderr.write(f"将 {len(failed_tracks)} 个DeepFace提取失败的tracks分配给 {speakers[0]['speaker_id']}\n")
	
	# 打印聚类详细信息
	sys.stderr.write(f"DeepFace聚类详情: 尝试了{len(thresholds_to_try)}种阈值, 最佳聚类数: {best_n_clusters}\n")
	for i, speaker in enumerate(speakers):
		track_list = speaker['tracks']
		sys.stderr.write(f"  {speaker['speaker_id']}: tracks {track_list}\n")
	
	return speakers

def generate_speaker_json(tracks, scores, features, args, output_path):
	"""
	生成每一帧的说话人标签JSON文件
	Args:
		tracks: 人脸追踪结果
		scores: 说话检测分数
		features: 人脸特征向量
		args: 参数
		output_path: 输出JSON文件路径
	"""
	# 1. 聚类说话人
	sys.stderr.write("开始进行DeepFace说话人聚类...\n")
	speakers = cluster_speakers_by_deepface_features(features, threshold=args.speakerThreshold)
	sys.stderr.write(f"检测到 {len(speakers)} 个说话人\n")
	
	# 2. 创建track到说话人的映射
	track_to_speaker = {}
	for speaker in speakers:
		for track_idx in speaker['tracks']:
			track_to_speaker[track_idx] = speaker['speaker_id']
	
	# 3. 获取所有帧
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	total_frames = len(flist)
	
	sys.stderr.write(f"开始生成 {total_frames} 帧的说话人标签...\n")
	
	# 4. 生成每帧的说话人信息
	frame_speakers = []
	
	for fidx in tqdm.tqdm(range(total_frames), desc="生成JSON"):
		timestamp = fidx / 25.0
		active_speakers = []
		
		# 遍历所有tracks，找到在当前帧出现的
		for tidx, track in enumerate(tracks):
			score = scores[tidx]
			frame_list = track['track']['frame'].tolist()
			
			if fidx in frame_list:
				frame_idx_in_track = frame_list.index(fidx)
				speaker_id = track_to_speaker.get(tidx, f"Unknown_{tidx}")
				
				# 平滑处理：取前后几帧的平均分数
				start_idx = max(frame_idx_in_track - 2, 0)
				end_idx = min(frame_idx_in_track + 3, len(score))
				smoothed_score = numpy.mean(score[start_idx:end_idx])
				
				is_speaking = bool(smoothed_score > 0)
				confidence = float(smoothed_score)
				
				active_speakers.append({
					"speaker_id": speaker_id,
					"is_speaking": is_speaking,
					"confidence": confidence
				})
		
		frame_speakers.append({
			"frame_id": fidx,
			"timestamp": round(timestamp, 3),
			"active_speakers": active_speakers
		})
	
	# 5. 生成完整JSON
	result = {
		"video_info": {
			"fps": 25,
			"total_frames": total_frames,
			"duration": round(total_frames / 25.0, 3),
			"detected_speakers": [speaker['speaker_id'] for speaker in speakers],
			"total_tracks": len(tracks),
			"speaker_mapping": {
				speaker['speaker_id']: {
					"tracks": speaker['tracks'],
					"total_tracks": len(speaker['tracks'])
				} for speaker in speakers
			}
		},
		"frame_speakers": frame_speakers
	}
	
	# 6. 保存JSON文件
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, 'w', encoding='utf-8') as f:
		json.dump(result, f, indent=2, ensure_ascii=False)
	
	sys.stderr.write(f"说话人标签JSON文件已保存到: {output_path}\n")
	
	# 7. 打印说话人统计信息
	sys.stderr.write("="*50 + "\n")
	sys.stderr.write("说话人统计信息:\n")
	for speaker in speakers:
		track_count = len(speaker['tracks'])
		sys.stderr.write(f"  {speaker['speaker_id']}: {track_count} 个tracks\n")
	sys.stderr.write("="*50 + "\n")
	
	return result

def visualization(tracks, scores, args):
	# CPU: visulize the result for video format
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = numpy.mean(s)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)

def evaluate_col_ASD(tracks, scores, args):
	txtPath = args.videoFolder + '/col_labels/fusion/*.txt' # Load labels
	predictionSet = {}
	for name in {'long', 'bell', 'boll', 'lieb', 'sick', 'abbas'}:
		predictionSet[name] = [[],[]]
	dictGT = {}
	txtFiles = glob.glob("%s"%txtPath)
	for file in txtFiles:
		lines = open(file).read().splitlines()
		idName = file.split('/')[-1][:-4]
		for line in lines:
			data = line.split('\t')
			frame = int(int(data[0]) / 29.97 * 25)
			x1 = int(data[1])
			y1 = int(data[2])
			x2 = int(data[1]) + int(data[3])
			y2 = int(data[2]) + int(data[3])
			gt = int(data[4])
			if frame in dictGT:
				dictGT[frame].append([x1,y1,x2,y2,gt,idName])
			else:
				dictGT[frame] = [[x1,y1,x2,y2,gt,idName]]	
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Load files
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]				
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = numpy.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)]) # average smoothing
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
		if fidx in dictGT: # This frame has label
			for gtThisFrame in dictGT[fidx]: # What this label is ?
				faceGT = gtThisFrame[0:4]
				labelGT = gtThisFrame[4]
				idGT = gtThisFrame[5]
				ious = []
				for face in faces[fidx]: # Find the right face in my result
					faceLocation = [int(face['x']-face['s']), int(face['y']-face['s']), int(face['x']+face['s']), int(face['y']+face['s'])]
					faceLocation_new = [int(face['x']-face['s']) // 2, int(face['y']-face['s']) // 2, int(face['x']+face['s']) // 2, int(face['y']+face['s']) // 2]
					iou = bb_intersection_over_union(faceLocation_new, faceGT, evalCol = True)
					if iou > 0.5:
						ious.append([iou, round(face['score'],2)])
				if len(ious) > 0: # Find my result
					ious.sort()
					labelPredict = ious[-1][1]
				else:					
					labelPredict = 0
				x1 = faceGT[0]
				y1 = faceGT[1]
				width = faceGT[2] - faceGT[0]
				predictionSet[idGT][0].append(labelPredict)
				predictionSet[idGT][1].append(labelGT)
	names = ['long', 'bell', 'boll', 'lieb', 'sick', 'abbas'] # Evaluate
	names.sort()
	F1s = 0
	for i in names:
		scores = numpy.array(predictionSet[i][0])
		labels = numpy.array(predictionSet[i][1])
		scores = numpy.int64(scores > 0)
		F1 = f1_score(labels, scores)
		ACC = accuracy_score(labels, scores)
		if i != 'abbas':
			F1s += F1
			print("%s, ACC:%.2f, F1:%.2f"%(i, 100 * ACC, 100 * F1))
	print("Average F1:%.2f"%(100 * (F1s / 5)))	  

# Main function
def main():
	# This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
	# ```
	# .
	# ├── pyavi
	# │   ├── audio.wav (Audio from input video)
	# │   ├── video.avi (Copy of the input video)
	# │   ├── video_only.avi (Output video without audio)
	# │   └── video_out.avi  (Output video with audio)
	# ├── pycrop (The detected face videos and audios)
	# │   ├── 000000.avi
	# │   ├── 000000.wav
	# │   ├── 000001.avi
	# │   ├── 000001.wav
	# │   └── ...
	# ├── pyframes (All the video frames in this video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...	
	# └── pywork
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     ├── scores.pckl (ASD result)
	#     └── tracks.pckl (face tracking result)
	# ```

	# Initialization 
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
	args.pycropPath = os.path.join(args.savePath, 'pycrop')
	if os.path.exists(args.savePath):
		rmtree(args.savePath)
	os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
	os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
	os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

	# Extract video
	args.videoFilePath = os.path.join(args.pyaviPath, 'video.avi')
	# If duration did not set, extract the whole video, otherwise extract the video from 'args.start' to 'args.start + args.duration'
	if args.duration == 0:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" % \
			(args.videoPath, args.nDataLoaderThread, args.videoFilePath))
	else:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" % \
			(args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the video and save in %s \r\n" %(args.videoFilePath))
	
	# Extract audio
	args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
	command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
		(args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

	# Extract the video frames
	command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" % \
		(args.videoFilePath, args.nDataLoaderThread, os.path.join(args.pyframesPath, '%06d.jpg'))) 
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the frames and save in %s \r\n" %(args.pyframesPath))

	# Scene detection for the video frames
	scene = scene_detect(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))	

	# Face detection for the video frames
	faces = inference_video(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

	# Face tracking
	allTracks, vidTracks = [], []
	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
			allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))

	# Face clips cropping
	for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
		vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)

	# Active Speaker Detection
	files = glob.glob("%s/*.avi"%args.pycropPath)
	files.sort()
	
	if args.generateJson:
		# Use the DeepFace enhanced function that extracts features
		scores, features = evaluate_network_with_deepface_features(files, args)
		# Save features for potential future use
		featuresPath = os.path.join(args.pyworkPath, 'deepface_features.pckl')
		with open(featuresPath, 'wb') as fil:
			pickle.dump(features, fil)
		sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " DeepFace features extracted and saved in %s \r\n" %args.pyworkPath)
	else:
		# Use the original function
		scores = evaluate_network(files, args)
		features = None
	
	savePath = os.path.join(args.pyworkPath, 'scores.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(scores, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

	if args.evalCol == True:
		evaluate_col_ASD(vidTracks, scores, args) # The columnbia video is too big for visualization. You can still add the `visualization` funcition here if you want
		quit()
	else:
		# Generate JSON file if requested
		if args.generateJson and features is not None:
			# 只影响json输出路径
			if not os.path.isabs(args.jsonOutputPath):
				json_output_path = os.path.join(args.videoFolder, args.jsonOutputPath)
			else:
				json_output_path = args.jsonOutputPath
			generate_speaker_json(vidTracks, scores, features, args, json_output_path)
		
		# Visualization, save the result as the new video	
		visualization(vidTracks, scores, args)	

if __name__ == '__main__':
    main()
