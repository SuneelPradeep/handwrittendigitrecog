import React from 'react'
import DrawingCanvas from './components/DrawingCanvas'

const App = () => {
  return (
    <section className='w-full text-white h-full mx-4 my-8 text-center '>
       <h1 className='mx-4'> Hand-Written Digit Recognition </h1>
       <DrawingCanvas />
    </section>
  )
}

export default App
