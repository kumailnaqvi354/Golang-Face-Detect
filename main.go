package main

import (
	"fmt"
	"gocv.io/x/gocv"
	"image"
)

func main() {

	// Load the image
	img := gocv.IMRead("/home/kumail/Projects/largest-face-golang/sample.jpg", gocv.IMReadColor)
	defer img.Close()

	// Load a pre-trained face detection model
	faceCascade := gocv.NewCascadeClassifier()
	faceCascade.Load("haarcascade_frontalface_default.xml") // You'll need to provide the XML file.

	//if faceCascade {
	//	panic("Error reading haarcascade_frontalface_default.xml")
	//}

	// Detect faces in the image
	faces := faceCascade.DetectMultiScale(img)

	if len(faces) == 0 {
		println("No faces found")
		return
	}

	// Find the largest face
	largestFace := image.Rect(0, 0, 0, 0)
	maxArea := 0

	for _, face := range faces {
		area := face.Dx() * face.Dy()
		if area > maxArea {
			largestFace = face
			maxArea = area
		}
	}
	x, y, w, h := largestFace.Min.X, largestFace.Min.Y, largestFace.Dx(), largestFace.Dy()

	fmt.Println("Debug X", x)
	fmt.Println("Debug X", y)
	fmt.Println("Debug X", w)
	fmt.Println("Debug X", h)

	//// Draw a rectangle around the largest face
	//gocv.Rectangle(&img, largestFace, color.RGBA{255, 0, 0, 0}, 2)
	//
	//// Display the image with the largest face highlighted
	//window := gocv.NewWindow("Largest Face")
	//defer window.Close()

	//window.IMShow(img)
	//window.WaitKey(0)
}
