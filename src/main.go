// Code snippets take from this tutorial: https://towardsdatascience.com/golang-for-machine-learning-bd4bb84594ee
// Same as here with minor changes: https://dev.to/salah856/build-a-simple-knn-model-using-golang-34db

package main

import (
	"fmt"
	// "log"
	// "os"

	// "github.com/go-gota/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func check(err error, s string) {
	if err != nil {
		panic(fmt.Sprintf("%s: %s", s, err.Error()))
	}
}

func main() {
	// irisCsv, err := os.Open("./data/iris_headers.csv")
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// df := dataframe.ReadCSV(irisCsv)
	// fmt.Println(df)

	fmt.Println("Load our csv data")
	rawData, err := base.ParseCSVToInstances("./data/iris_headers.csv", true)
	check(err, "Problem loading csv file")

	fmt.Println("Initialize our KNN classifier")
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	fmt.Println("Perform a training-test split")

	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	cls.Fit(trainData)

	fmt.Println("Calculate the euclidian distance and return the most popular label")

	predictions, err := cls.Predict(testData)
	check(err, "Problem calculating predictions")

	fmt.Println(predictions)

	fmt.Println("Print our summary metrics")
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)

	check(err, "Problem calculating confusion matrix")

	fmt.Println(evaluation.GetSummary(confusionMat))
}
