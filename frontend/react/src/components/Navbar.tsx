import React from 'react';
import { Link } from 'react-router-dom';
import {
  SignedIn,
  SignedOut,
  UserButton,
  SignInButton,
} from '@clerk/clerk-react';

interface NavbarProps {
  activePage?: string;
}

const Navbar: React.FC<NavbarProps> = ({ activePage = 'Dashboard' }) => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="app-title">Real Estate Investment Risk Detector</span>
      </div>

      <div className="navbar-menu">
        <Link to="/" className={`nav-item ${activePage === 'Dashboard' ? 'active' : ''}`}>
          Dashboard
        </Link>
        <Link to="/analytics" className={`nav-item ${activePage === 'Analytics' ? 'active' : ''}`}>
          Analytics
        </Link>
        <Link to="/properties" className={`nav-item ${activePage === 'Properties' ? 'active' : ''}`}>
          Properties
        </Link>
        <Link to="/reports" className={`nav-item ${activePage === 'Reports' ? 'active' : ''}`}>
          Reports
        </Link>
      </div>

      <div className="navbar-user">
        <SignedOut>
          <SignInButton />
        </SignedOut>

        <SignedIn>
          <UserButton
            afterSignOutUrl="/"
            appearance={{
              elements: {
                userButtonAvatarBox: "w-10 h-10 rounded-full overflow-hidden",
              },
            }}
          />
        </SignedIn>
      </div>
    </nav>
  );
};

export default Navbar;