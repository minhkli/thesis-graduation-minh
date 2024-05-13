import os
import cv2
from roboflow import Roboflow

rf = Roboflow(api_key="No9f9LhMpfr43cgmF4Qc")

workspace = rf.workspace("minhph")

project = workspace.project("agri-hoqax")
version = project.version(2)
model = version.model

prediction = model.predict("E:/Code/final_project-main/crop1/2019_0101_002305_102_JPG.jpg")

prediction.plot()
prediction.save(output_path="predictions1.jpg")