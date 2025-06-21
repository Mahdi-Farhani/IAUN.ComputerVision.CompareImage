
using IAUN.ComputerVision.FinalProject;
Console.WriteLine("IAUN COMPUTER VISION Final Project");
Console.WriteLine(new string('-', 50));
var mainImgPath = @"Images\612.jpg";
var folderPath = @"Test_Images";

List<int> distances = [1, 1];
Console.Write("Please enter prefered angle 1 (0,45,90,135):");
var firstAngleString = Console.ReadLine();
Console.Write("Please enter prefered angle 2 (0,45,90,135):");
var secondAngleString = Console.ReadLine();
var firstAngle = 0;
var secondAngle = 45;
if (!string.IsNullOrEmpty(firstAngleString))
{
    firstAngle = Convert.ToInt32(firstAngleString);
}
if (!string.IsNullOrEmpty(secondAngleString))
{
    firstAngle = Convert.ToInt32(secondAngle);
}

List<int> anglesDeg = [firstAngle, secondAngle];



var (bestName, dists) = ImageProcessor.CompareImages(mainImgPath, folderPath, distances, anglesDeg);

Console.WriteLine("the main image is similar to: " + bestName);

var files = Directory.GetFiles(folderPath);
for (int i = 0; i < files.Length; i++)
{
    Console.WriteLine(
        Path.GetFileName(files[i]) + ": " +
        dists[i].ToString("F4")
    );
}