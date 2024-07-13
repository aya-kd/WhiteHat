import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import './App.css';
import Lending from './pages/lending page/Lending';
import Scan from './pages/scan/Scan'
const App = () => {
  return (
    <Router>
    <Switch>
      <Route exact path="/">
        <Lending />
      </Route>
      <Route path="/scan">
        <Scan />
      </Route>
    </Switch>
  </Router>
);
};

export default App