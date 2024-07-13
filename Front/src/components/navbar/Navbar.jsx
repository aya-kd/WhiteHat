import React from 'react'
import './navbar.css'
import Group4 from  "../../assets/Group4.png"

const Navbar = () => {
  return (
    <div className="navbar"> 
       <div className="navbar_logo">
        <img src={Group4} alt="Group4" />
        <p>WhiteHat</p>
       </div>
      <div className="navbar_button">
        <button type="button" className='navbar_button' >SCAN</button>
      </div>
    </div>
   
  )
}

export default Navbar