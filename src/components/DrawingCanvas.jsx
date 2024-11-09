import axios from 'axios'
import React, { useEffect, useRef, useState } from 'react'

const DrawingCanvas = () => {
    const canvasRef = useRef(null)
    const [isDrawing,setIsDrawing] = useState(false)
   const [prediction,setPrediction] = useState(null)
   
   useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas dimensions
    canvas.width = 400; // or any desired width
    canvas.height = 300; // or any desired height
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height); // Fill the canvas with black

    // Optional: Set some default styles
    ctx.strokeStyle = 'white'; // Set default stroke color
    ctx.lineWidth = 2; // Set default line width
}, []);

   const startDrawing  = (e)=>{
      // e.preventDefault()
      // e.stopPropagation()
    setIsDrawing(true)
    const ctx = canvasRef.current.getContext('2d')
    ctx.beginPath()
    ctx.moveTo(e.nativeEvent.offsetX-50, e.nativeEvent.offsetY)
   }
   const draw = (e)=>{
      // e.preventDefault()
      // e.stopPropagation()
    if(!isDrawing) return;
    const ctx = canvasRef.current.getContext('2d')
    ctx.lineTo(e.nativeEvent.offsetX-50, e.nativeEvent.offsetY)
    ctx.stroke()
   }
   
   const clearCanvas = ()=>{
    const ctx = canvasRef.current.getContext('2d')
    ctx.clearRect(0,0,canvasRef.current.width,canvasRef.current.height)
    setPrediction(null)
   }
   const getCurrentCanvasImage = ()=>{
      const image = canvasRef.current?.toDataURL('image/png')
      console.log('the image is',image);
      
      return image;
      
   }
//    const getCurrentCanvasImage = () => {
//       const canvas = canvasRef.current;
//       const context = canvas.getContext('2d');
      
//       // Get image data from the canvas
//       const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
//       const data = imageData.data;
  
//       // Invert colors: white to black, black to white
//       for (let i = 0; i < data.length; i += 4) {
//           // Assuming the canvas is in RGBA format
//           // Invert the RGB values (ignoring alpha)
//           data[i] = 255 - data[i];     // Red
//           data[i + 1] = 255 - data[i + 1]; // Green
//           data[i + 2] = 255 - data[i + 2]; // Blue
//           // Alpha remains unchanged
//       }
  
//       // Put the modified data back to the context
//       context.putImageData(imageData, 0, 0);
  
//       // Now convert the modified canvas to a PNG data URL
//       const image = canvas.toDataURL('image/png');
//       console.log('The modified image is', image);
      
//       return image;
//   };
// const getCurrentCanvasImage = () => {
//    const canvas = canvasRef.current;
//    const context = canvas.getContext('2d');

//    // Get image data from the canvas
//    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
//    const data = imageData.data;

//    // Create a new imageData for the inverted colors
//    const invertedData = context.createImageData(canvas.width, canvas.height);

//    // Invert colors
//    for (let i = 0; i < data.length; i += 4) {
//        // Calculate the grayscale value
//        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;

//        // Set the inverted color (white becomes black and vice versa)
//        invertedData.data[i] = 255 - avg;     // Red
//        invertedData.data[i + 1] = 255 - avg; // Green
//        invertedData.data[i + 2] = 255 - avg; // Blue
//        invertedData.data[i + 3] = data[i + 3]; // Alpha
//    }

//    // Put the modified data back to the context
//    context.putImageData(invertedData, 0, 0);

//    // Now convert the modified canvas to a PNG data URL
//    const image = canvas.toDataURL('image/png');
//    console.log('The modified image is', image);
   
//    return image;
// };

  
  
   
   const stopDrawing = ()=>{
      setIsDrawing(false)
      const ctx = canvasRef.current.getContext('2d')
      ctx.closePath()
   }
   // useEffect(()=>{
   //      const fetchData =async ()=> { 
   //    const res = await axios.get("/.netlify/functions/predict/")
   //     //  console.log('the res is',res.data);
   //       }
   //     const data = fetchData()
      
        
   // },[])

   const handlePredict = async()=>{
      try {
      const imageData = getCurrentCanvasImage()
       
       const res = await axios.post("/.netlify/functions/predict",{ image : imageData})
      //const res = await axios.get("http://127.0.0.1:5000/random-predict")
      console.log('the res is',res?.data);
       setPrediction(res?.data)
      } catch (error) {
       
      }
}
   

  return (
    <div className='my-6 mx-8 flex flex-col items-center text-center justify-center'> 
     <h3 className='my-4 truncate'>The Predicted Number is <strong className='font-bold text-3xl mx-2'>{prediction?.prediction === 0 ? 0 : prediction?.prediction || ''}</strong> 
      Confidence is <strong className='font-bold text-3xl mx-2'>{prediction?.confidence || ''}</strong></h3>
      <canvas ref={canvasRef} className='mx-4 rounded-xl cursor-text bg-black text-white lg:w-[40rem] lg:h-[20rem] border-2 border-blue-500' 
      onMouseDown={startDrawing} onMouseMove={draw} onMouseUp={stopDrawing} onMouseLeave={stopDrawing}
      onTouchStart={startDrawing} onTouchEnd={stopDrawing} onTouchMove={draw} />
      <span className='my-2 gap-2 flex'> <button onClick={handlePredict}
       className='outline-none text-white bg-gradient-to-br from-black to-blue-400'> Predict</button>
       <button className='outline-none text-white bg-gradient-to-tr from-gray-800  to-blue-400' 
       onClick={clearCanvas}> Clear</button> </span>
    </div>
  )
}

export default DrawingCanvas
