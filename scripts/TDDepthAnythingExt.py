"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF

# Helpers
import gc

# Threading lib
import threading
import numpy as np
import logging
import os
import sys

logger = logging.getLogger('TDAppLogger')
try:
	from transformers import AutoImageProcessor, AutoModelForDepthEstimation
	import torch
except ImportError as e:
	logger.error(f'TDDepthAnything - An error occured trying to import some of the required libraries. Make sure that the environment is setup properly.')
	logger.error(f'TDDepthAnything - {e}')
	logger.error(f'TDDepthAnything - If you are using a custom python environment, make sure that the following packages are installed: transformers, torch')
except Exception as e:
	logger.error(f'TDDepthAnything - An error occured trying to import some of the required libraries. Make sure that the environment is setup properly.')
	logger.error(f'TDDepthAnything - {e}')

class TDDepthAnythingExt:
	"""
	TDDepthAnythingExt description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		
		# Utilities
		self.Logger = self.ownerComp.op('logger')
		self.SafeLogger = self.Logger.Logger
		self.ThreadManager = op.TDResources.ThreadManager
		self.InputImage = self.ownerComp.op('inputImage')
		self.ScriptBuffer = self.ownerComp.op('script1')

		self.DepthAnythingLock = threading.Lock()

		# ML Generics
		self.Model = None
		self.ImageProc = None
		
		self.NpDepth = np.random.randint(0, high=255, size=(720, 1280, 4), dtype='uint16')

		self.IsReady = False # Using IsReady to prevent inference when model is already running.
		
		self.Device ='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
		
		self.Logger.Debug('TDDepthAnything init done.', self)

	def __delTD__(self):
		# Called when the extension is deleted. Use this to clean up any resources.
		self.UnloadModelThreaded()
		self.Logger.Debug('TDDepthAnything __delTD__ done.', self)		

	def onInitTD(self):
		self.ScriptBuffer.copyNumpyArray(self.NpDepth)

	def PushToModel(self, model):
		with self.DepthAnythingLock:
			self.Model = model

	def GetModel(self):
		with self.DepthAnythingLock:
			return self.Model

	def PushToImageProc(self, imageProc):
		with self.DepthAnythingLock:
			self.ImageProc = imageProc
	
	def GetImageProc(self):
		with self.DepthAnythingLock:
			return self.ImageProc

	def SetNpDepth(self, npDepth):
		with self.DepthAnythingLock:
			self.NpDepth = npDepth

	def GetNpDepth(self):
		with self.DepthAnythingLock:
			return self.NpDepth

	def LoadModel(self, modelPath:str, checkpointsCache:str=None):
		currentTDThread = threading.current_thread()
		
		currentTDThread.SetProgressSafe(.0)
		infoDict = {
				'id': str(currentTDThread.ident),
				'name': currentTDThread.name,
				'messageType': 'TD_TM_TaskProgressUpdate',
				'progress': float(currentTDThread.Progress),
				'state': 'Processing'
		}
		currentTDThread.InfoQueue.put(infoDict)
		
		image_processor = AutoImageProcessor.from_pretrained(modelPath, cache_dir=checkpointsCache)
		
		currentTDThread.SetProgressSafe(.15)
		infoDict = {
				'id': str(currentTDThread.ident),
				'name': currentTDThread.name,
				'messageType': 'TD_TM_TaskProgressUpdate',
				'progress': float(currentTDThread.Progress),
				'state': 'Processing'
		}
		currentTDThread.InfoQueue.put(infoDict)
		
		model = AutoModelForDepthEstimation.from_pretrained(modelPath, cache_dir=checkpointsCache)
		
		currentTDThread.SetProgressSafe(.2)
		infoDict = {
				'id': str(currentTDThread.ident),
				'name': currentTDThread.name,
				'messageType': 'TD_TM_TaskProgressUpdate',
				'progress': float(currentTDThread.Progress),
				'state': 'Processing'
		}
		currentTDThread.InfoQueue.put(infoDict)

		model.to(device=self.Device)
		
		currentTDThread.SetProgressSafe(.6)
		infoDict = {
				'id': str(currentTDThread.ident),
				'name': currentTDThread.name,
				'messageType': 'TD_TM_TaskProgressUpdate',
				'progress': float(currentTDThread.Progress),
				'state': 'Processing'
		}
		currentTDThread.InfoQueue.put(infoDict)

		self.PushToImageProc(image_processor)
		
		currentTDThread.SetProgressSafe(.9)
		infoDict = {
				'id': str(currentTDThread.ident),
				'name': currentTDThread.name,
				'messageType': 'TD_TM_TaskProgressUpdate',
				'progress': float(currentTDThread.Progress),
				'state': 'Processing'
		}
		currentTDThread.InfoQueue.put(infoDict)

		self.PushToModel(model)
		
		currentTDThread.SetProgressSafe(1.0)
		infoDict = {
				'id': str(currentTDThread.ident),
				'name': currentTDThread.name,
				'messageType': 'TD_TM_TaskProgressUpdate',
				'progress': float(currentTDThread.Progress),
				'state': 'Processing'
		}
		currentTDThread.InfoQueue.put(infoDict)

	def LoadModelSuccess(self):
		model = self.GetModel()
		self.IsReady = True
		self.Logger.Info(f'The model {model} is loaded and ready for use.')	

	def LoadModelExcept(self, *args):
		self.SafeLogger.error(f'An error occured while trying to load the model, see thread exception for details. {args}')	

	def LoadModelRefresh(self):
		self.Logger.Debug(f'Model is loading, please wait.')
		return

	def LoadModelThreaded(self):
		modelName = 'Depth-Anything-V2-Small-hf'
		modelPath = f'depth-anything/{modelName}'
		checkpointsCache = f'{project.folder}/checkpoints/'
		
		self.SafeLogger.debug(f'Downloading model from HuggingFace or fetching from cache: {modelName}')
		
		myTDTask = self.ThreadManager.TDTask(
			target=self.LoadModel,
			SuccessHook=self.LoadModelSuccess,
			ExceptHook=self.LoadModelExcept,
			RefreshHook=self.LoadModelRefresh,
			args=(modelPath, checkpointsCache)
		)
		self.ThreadManager.EnqueueTask(myTDTask)
		self.SafeLogger.debug('Model thread started')

	def UnloadModel(self):
		try:
			model = self.GetModel()
			imageProc = self.GetImageProc()
			if model:
				self.PushToModel(None)
			
			if imageProc:
				self.PushToImageProc(None)

			gc.collect()
			torch.cuda.empty_cache()
		
		except:
			self.SafeLogger.debug('An error occurred while unloading the model.')
	
	def UnloadModelThreaded(self):
		myTDTask = self.ThreadManager.TDTask(target=self.UnloadModel)
		self.ThreadManager.EnqueueTask(myTDTask)
		self.SafeLogger.debug(myTDTask)

	def DepthInference(self, image, device):
		self.SafeLogger.debug('Starting Inference...')
		
		imageProc = self.GetImageProc()
		model = self.GetModel()

		if imageProc and model:
			# Default npArray shape from script TOP is (h, w, 4), we are getting rid of the alpha channel.
			image = self.PreprocessTDNpArray(image)
			res = image.shape[:2]
			
			inputs = imageProc(images=image, return_tensors="pt")
			inputs = {k: v.to(device) for k, v in inputs.items()}

			with torch.no_grad():
				outputs = model(**inputs)

			# interpolate to original size
			# We are downscaling so that it doesn't fill memory
			prediction = imageProc.post_process_depth_estimation(
				outputs,
				target_sizes=[res],
			)

			prediction = prediction[0]["predicted_depth"]
			buffer = self.PostprocessPrediction(prediction, res)
			self.SetNpDepth(buffer)

			self.SafeLogger.debug(f'Finished Inference... NpDepth = {self.NpDepth}')

		else:
			debug('The Image Pre-Processor and/or the model is/are not initialized, aborting.')
	
	def PreprocessTDNpArray(self, image):
		image = image[:, :, :3]  # Remove alpha channel
		image = image[:, :, ::-1]  # Convert BGR to RGB
		image = image.astype(np.float32) * 255
		return image

	def PostprocessPrediction(self, prediction, res):
		prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min())  # Normalize to [0, 1]
		prediction *= 65535  # Scale to 16-bit
		prediction = prediction.cpu().numpy().astype(np.uint16)

		# Create a buffer with 4 channels
		buffer = np.zeros((res[0], res[1], 4), dtype=np.uint16)
		buffer[:, :, 0] = prediction  # Assign depth to the first channel
		return buffer

	def DepthInferenceSuccess(self):
		self.IsReady = True
		self.ScriptBuffer.copyNumpyArray(self.GetNpDepth())

	def DepthInferenceExcept(self, *args):
		self.SafeLogger.error(f'The inference failed. See exception details. {args}')
		
	def DepthInferenceRefresh(self):
		self.Logger.Info('The inference is running, please wait.')

	def DepthInferenceThreaded(self):
		if self.IsReady:
			self.IsReady = False
			image = self.InputImage.numpyArray(delayed=False, writable=False)
			myTDTask = self.ThreadManager.TDTask(
				target=self.DepthInference,
				SuccessHook=self.DepthInferenceSuccess,
				ExceptHook=self.DepthInferenceExcept,
				RefreshHook=self.DepthInferenceRefresh,
				args=(image,self.Device)
			)			
			self.ThreadManager.EnqueueTask(myTDTask)
		
		else:
			if self.GetModel():
				self.Logger.Debug('Inference is already running, please wait.')
				return

			self.Logger.Debug('The model is not loaded, please load the model first.')
			return

	"""
	Parameter callbacks
	"""
	def OnPulseLoadmodel(self, par):
		self.LoadModelThreaded()
		self.Logger.Debug('Loading Model Triggered')
	
	def OnPulseUnloadmodel(self, par):
		self.UnloadModelThreaded()
		self.Logger.Debug('Unloading Model')

	def OnPulseTriggerinference(self, par):
		self.DepthInferenceThreaded()
		self.Logger.Debug('Running inference')

	def OnPulseReset(self, par):
		self.UnloadModelThreaded()
		self.SetNpDepth(np.random.randint(0, high=255, size=(2, 2, 4), dtype='uint16'))
		self.ScriptBuffer.copyNumpyArray(self.GetNpDepth())
		self.IsReady = False		
		return