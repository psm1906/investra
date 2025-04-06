// src/components/Sidebar.tsx

import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  SignedIn,
  SignedOut,
  UserButton,
  SignInButton,
  SignOutButton,
} from '@clerk/clerk-react';
import logo from '../assets/images/logo.png';

const Sidebar: React.FC = () => {
  return (
    <div className="sidebar">
      {/* Top Logo */}
      <div className="logo-container">
        <NavLink to="/" end>
          <img src={logo} alt="Logo" />
        </NavLink>
      </div>
      
      {/* User Authentication Section */}
      <div className="user-auth flex items-center justify-center mb-4">
        <SignedIn>
          <div className="flex items-center space-x-4">
            <UserButton
              afterSignOutUrl="/"
              appearance={{
                elements: {
                  userButtonAvatarBox: "w-[50px] h-[50px] p-4 rounded-full overflow-hidden",
                },
              }}
            />
            <SignOutButton
              afterSignOutUrl="/"
              className="menu-item signin-button"
            />
          </div>
        </SignedIn>
        <SignedOut>
          <SignInButton className="menu-item signin-button" />
        </SignedOut>
      </div>

      {/* Sidebar Menu */}
      <div className="sidebar-menu">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `menu-item ${isActive ? 'active' : ''}`
          }
        >
          <div className="menu-icon property-icon"></div>
          <span>Dashboard</span>
        </NavLink>

        <NavLink
          to="/properties"
          className={({ isActive }) =>
            `menu-item ${isActive ? 'active' : ''}`
          }
        >
          <div className="menu-icon dashboard-icon"></div>
          <span>Properties</span>
        </NavLink>

        <NavLink
          to="/analytics"
          className={({ isActive }) =>
            `menu-item ${isActive ? 'active' : ''}`
          }
        >
          <div className="menu-icon insights-icon"></div>
          <span>Analytics</span>
        </NavLink>

        <NavLink
          to="/reports"
          className={({ isActive }) =>
            `menu-item ${isActive ? 'active' : ''}`
          }
        >
          <div className="menu-icon history-icon"></div>
          <span>Reports</span>
        </NavLink>
      </div>
    </div>
  );
};

export default Sidebar;