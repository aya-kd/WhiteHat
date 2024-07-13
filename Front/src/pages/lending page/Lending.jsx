import React from 'react'
import './lending.css';
import { Link } from 'react-router-dom';
import Group4 from  "../../assets/Group4.png"
import Group5 from  "../../assets/Group5.png"


const Lending = () => {
  return (
    <div className='lending'>
      <div className='lending_navbar'>
        <img src={Group4} alt="group4" />
        <p>WhiteHat</p>
      </div>
      <div className='lending_content' >
        <div className='lending_content_text'>
          <h1>Your WhiteHat for all audits</h1>
          <p>Fortify your code and safeguard your assets. <br />
            Bring your WhiteHat for all of your audits.</p>
            <Link to="/scan">
             <button type="button">SCAN NOW</button>
            </Link>
          
        </div>
        <div className='lending_content_image'>
          <img src={Group5} alt="Group5" />
        </div>
      </div>

      
    </div>
  )
}

export default Lending