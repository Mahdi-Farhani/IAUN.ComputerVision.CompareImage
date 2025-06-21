// See https://aka.ms/new-console-template for more information
using IAUN.ComputerVision.FinalProject;
Console.WriteLine("Hello, World!");
string mainImgPath = @"Images\612.jpg";
string folderPath = @"E:\PHD\Codes_Practices\Computer Vision\wang_subset\wang_subset";

// پارامترهای GLCM
List<int> distances = new List<int> { 1, 2 };
List<int> anglesDeg = new List<int> { 0, 45, 90, 135 };

// مقایسه
var (bestName, dists) = ImageProcessor.CompareImages(mainImgPath, folderPath, distances, anglesDeg);

Console.WriteLine("نزدیک‌ترین تصویر: " + bestName);
// چاپ همهٔ نتایج
var files = Directory.GetFiles(folderPath);
for (int i = 0; i < files.Length; i++)
{
    Console.WriteLine(
        Path.GetFileName(files[i]) + ": " +
        dists[i].ToString("F4")
    );
}