package main

import (
	"fmt"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"
	// "github.com/sjwhitworth/golearn/base"
	// "github.com/sjwhitworth/golearn/evaluation"
	// "github.com/sjwhitworth/golearn/knn"
)

func main() {
	irisCsv, err := os.Open("../data/iris_headers.csv")
	if err != nil {
		log.Fatal(err)
	}

	df := dataframe.ReadCSV(irisCsv)
	head := df.Subset([]int{0, 3})
	fmt.Println(head)

	versicolorOnly := df.Filter(dataframe.F{
		Colname:    " Species",
		Comparator: "==",
		Comparando: "Iris-versicolor",
	})
	fmt.Println(versicolorOnly)
	attrFiltered := df.Select([]string{"Petal length", "Sepal length"})
	fmt.Println(attrFiltered)

}
