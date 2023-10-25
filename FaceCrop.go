package main

import (
	"fmt"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	"gocv.io/x/gocv"
	"image"
	"os"
)

func main() {
	// Load the image
	img := gocv.IMRead("/home/kumail/Projects/largest-face-golang/sample.jpg", gocv.IMReadColor)
	//if img == nil {
	//	fmt.Println("Failed to load image")
	//	os.Exit(1)
	//}

	// Convert the image to grayscale
	gray := gocv.NewMat()
	gocv.CvtColor(img, &gray, gocv.ColorBGRToGray)

	faceCascade := gocv.NewCascadeClassifier()
	faceCascade.Load("haarcascade_frontalface_default.xml")

	// Detect faces in the image

	faces := faceCascade.DetectMultiScale(img)
	if len(faces) == 0 {
		fmt.Println("No faces detected")
		os.Exit(1)
	}

	// Select the largest face
	largestFace := image.Rect(0, 0, 0, 0)
	maxArea := 0

	for _, face := range faces {
		area := face.Dx() * face.Dy()
		if area > maxArea {
			largestFace = face
			maxArea = area
		}
	}

	// Crop the largest face
	croppedFace := gocv.Mat.Region(img, image.Rect(0, 0, largestFace.Dx(), largestFace.Dy()))

	// Resize the cropped face to the model input size
	resizedFace := gocv.NewMat()

	gocv.Resize(croppedFace, &resizedFace, image.Point{224, 224}, 0, 0, gocv.InterpolationLinear)

	// Normalize the cropped face
	resizedFace.SubtractFloat(127.5)
	resizedFace.DivideFloat(127.5)

	// Convert the normalized face to a TensorFl	ow tensor
	tensor, _ := tensorflow.NewTensor(resizedFace)

	// Load the model
	model, _ := tensorflow.LoadSavedModel("eval_image_classifier", nil, nil)

	// Run the model on the input tensor
	//output, err := model.Session.Run(nil, []tensorflow.Tensor{tensor}, []string{"output"})
	output, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{model.Graph.Operation("input").Output(0): tensor},
		[]tensorflow.Output{model.Graph.Operation("output").Output(0)},
		nil,
	)

	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Get the largest face quants from the output tensor
	largestFaceQuants := output[0].Value().([]float32)

	// Print the largest face quants
	fmt.Println(largestFaceQuants)
}
